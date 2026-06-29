//! ChatGPT OAuth — issue / refresh / persist access tokens for a ChatGPT account.
//!
//! Tokens live in `~/.codex/auth.json` (the location shared with the official
//! Codex CLI), so logging in here also logs you in for the Codex CLI and vice
//! versa.

use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use base64::Engine;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

// OAuth endpoints / tuning values: implementation details, not public API.
pub(crate) const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
pub(crate) const ISSUER: &str = "https://auth.openai.com";
pub(crate) const TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
pub(crate) const DEVICE_USERCODE_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/usercode";
pub(crate) const DEVICE_POLL_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/token";
pub(crate) const DEVICE_REDIRECT_URI: &str = "https://auth.openai.com/deviceauth/callback";
pub(crate) const DEFAULT_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";
pub(crate) const REFRESH_SKEW_SECONDS: i64 = 120; // refresh this many seconds before expiry
pub(crate) const MAX_POLL_INTERVAL_SECS: u64 = 60;
/// Cap untrusted response bodies (e.g. error bodies) when reading, to prevent
/// memory blow-up from a hostile or buggy server.
pub(crate) const MAX_ERROR_BODY: usize = 8192;

const TRUSTED_HOST_SUFFIXES: &[&str] = &["chatgpt.com", "openai.com"];

#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("relogin required: {0}")]
    ReloginRequired(String),
    #[error("{0}")]
    Other(String),
}

/// True if `err` carries an `AuthError::ReloginRequired` (token expired/missing/
/// denied). Lets callers branch to `device_code_login()` and retry.
pub fn is_relogin_required(err: &anyhow::Error) -> bool {
    matches!(err.downcast_ref::<AuthError>(), Some(AuthError::ReloginRequired(_)))
}

// Debug is hand-rolled (below) to redact tokens / API key from logs and panics.
#[derive(Clone, Serialize, Deserialize)]
struct AuthFile {
    // A file lacking `tokens` (e.g. OPENAI_API_KEY-only) parses as empty rather
    // than failing, matching the "parses but lacks tokens -> Ok(None)" contract
    // and preserving keys written by other clients on first login.
    #[serde(default)]
    tokens: TokensFile,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    last_refresh: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", rename = "OPENAI_API_KEY")]
    openai_api_key: Option<String>,
    /// Preserve unknown fields so we don't clobber data written by other clients
    /// (e.g. the official Codex CLI).
    #[serde(flatten)]
    extra: serde_json::Map<String, serde_json::Value>,
}

// Redacted Debug so secrets don't print via `{:?}`.
impl std::fmt::Debug for AuthFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthFile")
            .field("tokens", &self.tokens)
            .field("last_refresh", &self.last_refresh)
            .field(
                "openai_api_key",
                &self.openai_api_key.as_ref().map(|_| "[redacted]"),
            )
            .field("extra", &self.extra)
            .finish()
    }
}

// Default + field defaults: a valid-JSON file with absent/partial token keys is
// treated as empty (not corrupt), so load returns Ok(None) and save preserves
// OPENAI_API_KEY/extra. Syntactically broken JSON still fails as corrupt.
#[derive(Clone, Default, Serialize, Deserialize)]
struct TokensFile {
    #[serde(default)]
    access_token: String,
    #[serde(default)]
    refresh_token: String,
    #[serde(flatten)]
    extra: serde_json::Map<String, serde_json::Value>,
}

impl std::fmt::Debug for TokensFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokensFile")
            .field("access_token", &"[redacted]")
            .field("refresh_token", &"[redacted]")
            .field("extra", &self.extra)
            .finish()
    }
}

// Runtime credentials, decoupled from the on-disk AuthFile format.
#[derive(Clone)]
pub struct CodexCredentials {
    pub access_token: String,
    pub refresh_token: String,
    pub base_url: String,
    pub last_refresh: Option<String>,
    /// True if auto-loaded from disk (`~/.codex/auth.json`). Such tokens are sent
    /// only to trusted hosts (or loopback) even when the
    /// `CODEX_ALLOW_INSECURE_BASE_URL` bypass is on, so a forgotten flag can't
    /// leak a real subscription token to an arbitrary host. Injected tokens: false.
    pub from_disk: bool,
}

// Redacted Debug: hide tokens, expose only routing/meta fields.
impl std::fmt::Debug for CodexCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CodexCredentials")
            .field("access_token", &"[redacted]")
            .field("refresh_token", &"[redacted]")
            .field("base_url", &self.base_url)
            .field("last_refresh", &self.last_refresh)
            .field("from_disk", &self.from_disk)
            .finish()
    }
}

impl CodexCredentials {
    /// JWT signature is NOT verified. This is only a backend routing hint, not an
    /// authorization decision — the backend authorizes via the Bearer header.
    pub fn chatgpt_account_id(&self) -> Option<String> {
        let claims = decode_jwt_claims(&self.access_token)?;
        let auth = claims.get("https://api.openai.com/auth")?;
        auth.get("chatgpt_account_id")?.as_str().map(String::from)
    }
}

// ──────────────────────────────────────────────────────────────────────
// Storage — ~/.codex/auth.json
// ──────────────────────────────────────────────────────────────────────

/// Absolute path to `~/.codex/auth.json`.
/// Precedence: `CODEX_HOME` env var → `dirs::home_dir()`. If both fail this
/// returns Err — we do NOT fall back to cwd, to avoid leaking secrets into a
/// random working directory (e.g. a git repo).
pub fn auth_path() -> Result<PathBuf> {
    if let Ok(custom) = std::env::var("CODEX_HOME") {
        let trimmed = custom.trim();
        if !trimmed.is_empty() {
            return codex_home_to_auth_path(trimmed);
        }
    }
    let home = dirs::home_dir().ok_or_else(|| {
        anyhow!(
            "cannot locate home directory (HOME unset?). \
             Set the CODEX_HOME environment variable to an explicit directory. \
             Fallback to cwd is disabled to avoid leaking secrets."
        )
    })?;
    Ok(home.join(".codex").join("auth.json"))
}

/// Convert a (trimmed, non-empty) `CODEX_HOME` value to the auth.json path.
/// Rejects relative paths: a value like `CODEX_HOME=.codex` would write tokens
/// into the cwd (e.g. a git repo) and risk committing secrets, contradicting the
/// disabled cwd fallback.
fn codex_home_to_auth_path(custom: &str) -> Result<PathBuf> {
    let dir = PathBuf::from(custom);
    if !dir.is_absolute() {
        bail!(
            "CODEX_HOME must be an absolute path (got `{custom}`). \
             A relative path would write tokens into the current working \
             directory (e.g. a git repo), risking secret leakage."
        );
    }
    Ok(dir.join("auth.json"))
}

/// Returns `Ok(None)` if the file is missing OR if it parses but lacks tokens
/// (compatible with the first-login flow). A file that exists but does NOT parse
/// is a corrupt token store and returns `Err` — this matches the save side, which
/// also refuses to touch a malformed file. (Previously load swallowed parse
/// errors as `Ok(None)`, so callers saw "no token, please log in" and then the
/// subsequent save failed with "corrupted" — load and save disagreed.)
pub fn load_codex_cli_tokens() -> Result<Option<CodexCredentials>> {
    let path = auth_path()?;
    if !path.is_file() {
        return Ok(None);
    }
    // Tighten group/other-accessible files to 0600 before reading.
    ensure_owner_only_perms(&path)?;
    let bytes = std::fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let file: AuthFile = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        // Parse failure = corrupt store. Report Err to match the save side;
        // swallowing as None causes "no token -> relogin -> save fails" disagreement.
        Err(e) => bail!(
            "{} exists but is not valid JSON (corrupt token store). \
             Back it up and remove it, then run device_code_login() again. Cause: {}",
            path.display(),
            e
        ),
    };
    if file.tokens.access_token.trim().is_empty() || file.tokens.refresh_token.trim().is_empty() {
        return Ok(None);
    }
    Ok(Some(CodexCredentials {
        access_token: file.tokens.access_token.trim().to_string(),
        refresh_token: file.tokens.refresh_token.trim().to_string(),
        base_url: default_base_url(),
        last_refresh: file.last_refresh,
        from_disk: true,
    }))
}

/// Atomic write (tmp file + rename). Refuses to overwrite an existing
/// malformed file — that protects fields other clients may have added (e.g.
/// `OPENAI_API_KEY` or unknown `extra` keys) from being clobbered.
///
/// NOTE: this is the **unlocked** primitive. It does NOT take the cross-process
/// auth-file lock, so it must only be called either (a) while already holding the
/// lock (as `resolve_inner` does) or (b) via [`save_codex_cli_tokens_locked`].
/// Calling it bare from a context that races refresh can clobber a freshly
/// rotated `refresh_token`.
///
/// pub(crate): the unlocked primitive is excluded from the public API; callers
/// must use `save_codex_cli_tokens_locked`.
pub(crate) fn save_codex_cli_tokens(creds: &CodexCredentials) -> Result<()> {
    let path = auth_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    // Read-modify-write to preserve existing fields, or start empty.
    let mut file: AuthFile = if path.is_file() {
        let bytes = std::fs::read(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        match serde_json::from_slice(&bytes) {
            Ok(v) => v,
            // Refuse to overwrite a corrupt file — would clobber other clients' fields.
            Err(e) => bail!(
                "refusing to write tokens because {} is corrupted \
                 (would clobber other clients' preserved fields). \
                 Back it up and remove it, then retry. Cause: {}",
                path.display(),
                e
            ),
        }
    } else {
        empty_auth_file()
    };
    file.tokens.access_token = creds.access_token.clone();
    file.tokens.refresh_token = creds.refresh_token.clone();
    file.last_refresh = Some(creds.last_refresh.clone().unwrap_or_else(now_iso));

    let serialized = serde_json::to_vec_pretty(&file)?;

    // Atomic write step 1: build a unique tmp name (pid + nanos). A fixed name in
    // a shared CODEX_HOME could collide with a pre-seeded loose-perm file or an
    // attacker symlink, defeating mode(0o600) or leaking tokens to the link
    // target. write_file_owner_only opens with create_new (O_EXCL), so it never
    // follows an existing file/link.
    let tmp_path = match path.file_name() {
        Some(name) => {
            let pid = std::process::id();
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let mut tmp_name = std::ffi::OsString::from(".");
            tmp_name.push(name);
            tmp_name.push(format!(".{pid}.{nanos}.tmp"));
            path.with_file_name(tmp_name)
        }
        None => bail!("auth_path has no filename component: {}", path.display()),
    };
    write_file_owner_only(&tmp_path, &serialized)
        .with_context(|| format!("failed to write {}", tmp_path.display()))?;
    // Atomic write step 2: rename, so no half-written file is ever visible.
    if let Err(e) = std::fs::rename(&tmp_path, &path) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(anyhow!(e)).with_context(|| {
            format!("failed to rename {} -> {}", tmp_path.display(), path.display())
        });
    }
    Ok(())
}

/// Locked public entry point for saving. Callers/login paths must use this to
/// serialize against concurrent refresh. Must NOT be called from a context that
/// already holds the lock (e.g. `resolve_inner`) — would deadlock on flock.
pub async fn save_codex_cli_tokens_locked(creds: &CodexCredentials) -> Result<()> {
    // 2-stage lock; drop order is reverse of declaration: flock first, then mutex.
    let _guard = refresh_lock().lock().await;
    let _flock = acquire_auth_file_lock().await?;
    save_codex_cli_tokens(creds)
}

#[cfg(unix)]
fn write_file_owner_only(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)   // O_EXCL: fail if it exists, never follow a pre-seeded file/symlink
        .mode(0o600)        // 0600 from creation time, no perm race
        .open(path)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    Ok(())
}

#[cfg(not(unix))]
fn write_file_owner_only(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    // Windows: we don't tighten the file ACL from code. Relies on the user's
    // profile directory inheriting a reasonable ACL — do NOT place CODEX_HOME
    // on a shared / world-readable directory.
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    Ok(())
}

// Tighten an existing token file to 0600 if group/other bits are set. We tighten
// (and warn) rather than hard-reject, so a file another tool created at 0644
// stays usable.
#[cfg(unix)]
fn ensure_owner_only_perms(path: &std::path::Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let meta = std::fs::metadata(path)
        .with_context(|| format!("failed to stat {}", path.display()))?;
    let mode = meta.permissions().mode();
    if mode & 0o077 != 0 {
        tracing::warn!(
            path = %path.display(),
            mode = format!("{:o}", mode & 0o777),
            "auth.json is group/other-accessible — tightening to 0600"
        );
        let mut perms = meta.permissions();
        perms.set_mode(0o600);
        std::fs::set_permissions(path, perms)
            .with_context(|| format!("failed to chmod {} to 0600", path.display()))?;
    }
    Ok(())
}

#[cfg(not(unix))]
fn ensure_owner_only_perms(_path: &std::path::Path) -> Result<()> {
    Ok(())
}

fn empty_auth_file() -> AuthFile {
    AuthFile {
        tokens: TokensFile {
            access_token: String::new(),
            refresh_token: String::new(),
            extra: Default::default(),
        },
        last_refresh: None,
        openai_api_key: None,
        extra: Default::default(),
    }
}

// ──────────────────────────────────────────────────────────────────────
// JWT expiry check
// ──────────────────────────────────────────────────────────────────────

// Decode the JWT payload (middle segment) to JSON. None on any failure.
fn decode_jwt_claims(token: &str) -> Option<serde_json::Value> {
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() < 2 {
        return None;
    }
    let mut payload = parts[1].to_string();
    // base64url length must be a multiple of 4; pad with '='.
    let padding = (4 - payload.len() % 4) % 4;
    payload.push_str(&"=".repeat(padding));
    let bytes = base64::engine::general_purpose::URL_SAFE
        .decode(payload.as_bytes())
        .ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Safe default: if we cannot decode the JWT, or the `exp` claim is missing,
/// or the system clock looks invalid, return `true` (treat as expiring). The
/// alternative — silently treating an unverifiable token as fresh — would
/// defer the failure to the next API call.
pub fn is_access_token_expiring(token: &str, skew_seconds: i64) -> bool {
    let Some(claims) = decode_jwt_claims(token) else {
        return true;
    };
    let Some(exp) = claims.get("exp").and_then(|v| v.as_f64()) else {
        return true;
    };
    let Ok(now_dur) = SystemTime::now().duration_since(UNIX_EPOCH) else {
        return true;
    };
    let now = now_dur.as_secs_f64();
    exp <= now + (skew_seconds.max(0) as f64)
}

// ──────────────────────────────────────────────────────────────────────
// Device code login (one-time)
// ──────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct DeviceCodeResp {
    user_code: String,
    device_auth_id: String,
    #[serde(default)]
    interval: Option<serde_json::Value>, // shape varies, so Value
}

#[derive(Deserialize)]
struct PollResp {
    authorization_code: String,
    code_verifier: String,
}

// Redacted Debug: auth code / PKCE verifier are secrets.
impl std::fmt::Debug for PollResp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PollResp")
            .field("authorization_code", &"[redacted]")
            .field("code_verifier", &"[redacted]")
            .finish()
    }
}

#[derive(Deserialize)]
struct TokenResp {
    access_token: String,
    refresh_token: Option<String>,
}

// Redacted Debug.
impl std::fmt::Debug for TokenResp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenResp")
            .field("access_token", &"[redacted]")
            .field(
                "refresh_token",
                &self.refresh_token.as_ref().map(|_| "[redacted]"),
            )
            .finish()
    }
}

/// Sign-in prompt shown during `device_code_login`. Non-CLI consumers can
/// receive this via [`device_code_login_with`] and render it in their own UI.
#[derive(Clone)]
pub struct DeviceCodePrompt<'a> {
    pub verification_url: &'a str,
    pub user_code: &'a str,
}

// Redacted Debug: hand-rolled so `user_code` doesn't leak into logs via `{:?}`.
// (Intentional display goes through the `user_code` field directly, not Debug.)
impl std::fmt::Debug for DeviceCodePrompt<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceCodePrompt")
            .field("verification_url", &self.verification_url)
            .field("user_code", &"[redacted]")
            .finish()
    }
}

/// Run device-code login, printing the URL/code to stdout (CLI default). To
/// intercept the output (GUI/TUI/bot), use [`device_code_login_with`].
pub async fn device_code_login() -> Result<CodexCredentials> {
    device_code_login_with(default_device_prompt).await
}

/// Default prompt: print the login URL/code to stdout (CLI).
fn default_device_prompt(p: &DeviceCodePrompt<'_>) -> Result<()> {
    println!("\nTo sign in with your ChatGPT account:\n");
    println!("  1. Open in your browser: \x1b[94m{}\x1b[0m", p.verification_url);
    println!("  2. Enter this code:      \x1b[94m{}\x1b[0m\n", p.user_code);
    println!("Waiting for sign-in... (Ctrl+C to cancel)");
    Ok(())
}

/// Callback variant of [`device_code_login`]: delivers the prompt via `on_prompt`
/// instead of stdout, called exactly once after code issuance and before polling.
///
/// If the callback returns `Err` (e.g. a bot fails to post the code), polling
/// never starts and the error surfaces — avoids 15 minutes of useless polling
/// when the user never received the code.
pub async fn device_code_login_with<F>(on_prompt: F) -> Result<CodexCredentials>
where
    F: FnOnce(&DeviceCodePrompt<'_>) -> Result<()>,
{
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()?;

    // 1) Request a device code.
    let resp = client
        .post(DEVICE_USERCODE_URL)
        .json(&serde_json::json!({ "client_id": CLIENT_ID }))
        .send()
        .await
        .context("failed to send device_code request")?;
    if !resp.status().is_success() {
        bail!("device_code request failed: HTTP {}", resp.status());
    }
    let device: DeviceCodeResp = resp
        .json()
        .await
        .context("failed to parse device_code response JSON")?;
    // Clamp the server-supplied poll interval into [3, MAX_POLL_INTERVAL_SECS]
    // so a hostile or buggy server cannot make us wait forever.
    let mut poll_interval = device
        .interval
        .as_ref()
        .and_then(|v| v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse().ok())))
        .unwrap_or(5)
        .clamp(3, MAX_POLL_INTERVAL_SECS);
    if device.user_code.is_empty() || device.device_auth_id.is_empty() {
        bail!("device_code response is missing required fields");
    }

    // Deliver the prompt; an Err here aborts before polling starts.
    let verification_url = format!("{ISSUER}/codex/device");
    on_prompt(&DeviceCodePrompt {
        verification_url: &verification_url,
        user_code: &device.user_code,
    })
    .context("failed to deliver device-code prompt to caller")?;

    let max_wait = Duration::from_secs(15 * 60);
    let start = std::time::Instant::now();
    let code_resp: PollResp = loop {
        if start.elapsed() > max_wait {
            bail!("sign-in not completed within 15 minutes");
        }
        tokio::time::sleep(Duration::from_secs(poll_interval)).await;
        // 2) Poll for completion.
        let poll = client
            .post(DEVICE_POLL_URL)
            .json(&serde_json::json!({
                "device_auth_id": device.device_auth_id,
                "user_code": device.user_code,
            }))
            .send()
            .await
            .context("failed to send device_code poll request")?;
        match poll.status() {
            StatusCode::OK => {
                break poll
                    .json()
                    .await
                    .context("failed to parse poll response JSON")?;
            }
            // This backend signals "not yet" with 403/404.
            StatusCode::FORBIDDEN | StatusCode::NOT_FOUND => continue,
            // Otherwise inspect the standard RFC 8628 error code; the backend may
            // return pending as 4xx + code, so don't bail unconditionally.
            other => {
                let body = bounded_text(poll, MAX_ERROR_BODY).await;
                let (code, msg) = parse_oauth_error(&body);
                match code.as_deref() {
                    Some("authorization_pending") => continue,
                    Some("slow_down") => {
                        poll_interval = (poll_interval + 5).min(MAX_POLL_INTERVAL_SECS);
                        continue;
                    }
                    Some("expired_token") => {
                        bail!("device code expired before sign-in completed; restart login");
                    }
                    Some("access_denied") => bail!("sign-in was denied"),
                    _ => match msg {
                        Some(m) => bail!("device poll error: HTTP {other}: {m}"),
                        None => bail!("device poll error: HTTP {other}"),
                    },
                }
            }
        }
    };

    // 3) Exchange the authorization code for tokens.
    let form = [
        ("grant_type", "authorization_code"),
        ("code", &code_resp.authorization_code),
        ("redirect_uri", DEVICE_REDIRECT_URI),
        ("client_id", CLIENT_ID),
        ("code_verifier", &code_resp.code_verifier),
    ];
    let token_resp = client
        .post(TOKEN_URL)
        .form(&form)
        .send()
        .await
        .context("failed to send token exchange request")?;
    if !token_resp.status().is_success() {
        bail!("token exchange failed: HTTP {}", token_resp.status());
    }
    let tokens: TokenResp = token_resp
        .json()
        .await
        .context("failed to parse token exchange response JSON")?;
    let access = tokens.access_token.trim().to_string();
    let refresh = tokens
        .refresh_token
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    if access.is_empty() || refresh.is_empty() {
        bail!("token exchange response is missing access/refresh");
    }

    let creds = CodexCredentials {
        access_token: access,
        refresh_token: refresh,
        base_url: default_base_url(),
        last_refresh: Some(now_iso()),
        from_disk: true,
    };
    // Save under the same lock as refresh; an unlocked save could race refresh's
    // read-modify-write and clobber a freshly rotated refresh_token.
    save_codex_cli_tokens_locked(&creds).await?;
    Ok(creds)
}

/// Convenience: return stored tokens if present, else run [`device_code_login`].
/// (A corrupt auth.json surfaces as Err — back up/remove and retry.) To intercept
/// the prompt, combine [`load_codex_cli_tokens`] + [`device_code_login_with`].
pub async fn ensure_logged_in() -> Result<CodexCredentials> {
    match load_codex_cli_tokens()? {
        Some(creds) => Ok(creds),
        None => device_code_login().await,
    }
}

// ──────────────────────────────────────────────────────────────────────
// Refresh
// ──────────────────────────────────────────────────────────────────────

// Exchange a refresh token for a new (access, refresh) pair.
// pub(crate): low-level primitive (no lock, no disk write); callers use
// `resolve_credentials`.
pub(crate) async fn refresh_tokens(refresh_token: &str) -> Result<(String, String)> {
    if refresh_token.trim().is_empty() {
        return Err(anyhow!(AuthError::ReloginRequired(
            "refresh_token is empty".into()
        )));
    }
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .build()?;
    let form = [
        ("grant_type", "refresh_token"),
        ("refresh_token", refresh_token),
        ("client_id", CLIENT_ID),
    ];
    let resp = client
        .post(TOKEN_URL)
        .form(&form)
        .send()
        .await
        .context("failed to send token refresh request")?;
    let status = resp.status();
    if !status.is_success() {
        let body = bounded_text(resp, MAX_ERROR_BODY).await;
        let (code, msg) = parse_oauth_error(&body);
        // These error codes or 401/403 mean the refresh token is unusable -> relogin.
        let relogin = matches!(
            code.as_deref(),
            Some("invalid_grant" | "invalid_token" | "invalid_request" | "refresh_token_reused")
        ) || matches!(status, StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN);
        let pretty = msg.unwrap_or_else(|| format!("token refresh failed: HTTP {status}"));
        if relogin {
            tracing::warn!(%status, "token refresh failed — relogin required");
            return Err(anyhow!(AuthError::ReloginRequired(pretty)));
        }
        tracing::warn!(%status, "token refresh failed");
        bail!(pretty);
    }
    let tr: TokenResp = resp
        .json()
        .await
        .context("failed to parse refresh response JSON")?;
    let access = tr.access_token.trim().to_string();
    if access.is_empty() {
        return Err(anyhow!(AuthError::ReloginRequired(
            "refresh response had no access_token".into()
        )));
    }
    // Use the rotated refresh token if returned, else keep the old one (servers
    // may omit it on rotation).
    let new_refresh = tr
        .refresh_token
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| refresh_token.trim().to_string());
    Ok((access, new_refresh))
}

/// Read up to `max_bytes` of a response body as a (lossy) String, to defang
/// malicious / huge error bodies before logging.
pub(crate) async fn bounded_text(resp: reqwest::Response, max_bytes: usize) -> String {
    use futures_util::StreamExt;
    let mut buf: Vec<u8> = Vec::new();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let Ok(bytes) = chunk else { break };
        let remain = max_bytes.saturating_sub(buf.len());
        if remain == 0 {
            break;
        }
        if bytes.len() <= remain {
            buf.extend_from_slice(&bytes);
        } else {
            buf.extend_from_slice(&bytes[..remain]);
            break;
        }
    }
    String::from_utf8_lossy(&buf).into_owned()
}

// Extract (error code, human message) from an OAuth error body. Both optional.
fn parse_oauth_error(body: &str) -> (Option<String>, Option<String>) {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(body) else {
        return (None, None);
    };
    // Shape 1: OpenAI {"error": {"code": "...", "message": "..."}}
    if let Some(err) = v.get("error").and_then(|e| e.as_object()) {
        let code = err
            .get("code")
            .or_else(|| err.get("type"))
            .and_then(|x| x.as_str())
            .map(String::from);
        let msg = err
            .get("message")
            .and_then(|x| x.as_str())
            .map(|s| format!("token refresh failed: {s}"));
        return (code, msg);
    }
    // Shape 2: plain OAuth {"error": "code", "error_description": "..."}
    if let Some(s) = v.get("error").and_then(|x| x.as_str()) {
        let desc = v
            .get("error_description")
            .or_else(|| v.get("message"))
            .and_then(|x| x.as_str())
            .map(|d| format!("token refresh failed: {d}"));
        return (Some(s.to_string()), desc);
    }
    (None, None)
}

// ──────────────────────────────────────────────────────────────────────
// Public entry point — usable credentials with auto-refresh
// ──────────────────────────────────────────────────────────────────────

/// Serialize concurrent refreshes within a single process. Cross-process races
/// are handled naturally by the OAuth server returning `refresh_token_reused`,
/// which surfaces as a `ReloginRequired` error to the user.
fn refresh_lock() -> &'static tokio::sync::Mutex<()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

/// Cross-process lock file path (`auth.json` -> `auth.json.lock`).
fn auth_lock_path() -> Result<PathBuf> {
    Ok(auth_path()?.with_extension("json.lock"))
}

/// File-lock guard. Dropping it closes the fd, which auto-releases the OS flock
/// (POSIX flock / Windows LockFileEx alike). The field exists only to keep the
/// fd alive for the guard's lifetime.
struct FileLockGuard(#[allow(dead_code)] std::fs::File);

/// Acquire a cross-process exclusive lock so concurrent refreshes across
/// processes sharing `~/.codex/auth.json` don't force a `refresh_token_reused`
/// logout. Uses try-lock + async sleep so blocking flock doesn't stall the
/// executor; a dead holder releases on fd close, so no permanent deadlock
/// (timeout is a safety net).
async fn acquire_auth_file_lock() -> Result<FileLockGuard> {
    use fs2::FileExt;
    let path = auth_lock_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let file = std::fs::OpenOptions::new()
        .create(true)
        // Lock file is used only as a flock handle; never touch its contents.
        .truncate(false)
        .read(true)
        .write(true)
        .open(&path)
        .with_context(|| format!("failed to open lock file {}", path.display()))?;
    let start = Instant::now();
    let mut logged_wait = false;
    loop {
        match file.try_lock_exclusive() {
            Ok(()) => return Ok(FileLockGuard(file)),
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                if !logged_wait {
                    tracing::debug!("waiting for cross-process auth.json lock");
                    logged_wait = true;
                }
                if start.elapsed() > Duration::from_secs(30) {
                    bail!("timed out waiting for auth file lock: {}", path.display());
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            Err(e) => {
                return Err(e).with_context(|| format!("failed to lock {}", path.display()));
            }
        }
    }
}

/// Shared impl: acquire lock -> reload tokens -> if `should_refresh` (judged on
/// the re-read tokens) refresh and save. Re-reading under the lock lets us skip
/// when another task/process already refreshed.
async fn resolve_inner(
    should_refresh: impl Fn(&CodexCredentials) -> bool,
) -> Result<CodexCredentials> {
    // 2-stage lock (in-process mutex + cross-process flock).
    // Drop order is reverse of declaration: flock first, then mutex.
    let _guard = refresh_lock().lock().await;
    let _flock = acquire_auth_file_lock().await?;
    let Some(mut creds) = load_codex_cli_tokens()? else {
        return Err(anyhow!(AuthError::ReloginRequired(format!(
            "no Codex token at {}. Call device_code_login() first.",
            auth_path()?.display()
        ))));
    };
    if should_refresh(&creds) {
        tracing::debug!("access token needs refresh — calling token endpoint");
        let (new_access, new_refresh) = refresh_tokens(&creds.refresh_token).await?;
        creds.access_token = new_access;
        creds.refresh_token = new_refresh;
        creds.last_refresh = Some(now_iso());
        save_codex_cli_tokens(&creds)?;
    }
    Ok(creds)
}

/// Main entry point: return ready-to-use credentials, refreshing if
/// `force_refresh` or the token is near expiry (proactive path).
pub async fn resolve_credentials(force_refresh: bool) -> Result<CodexCredentials> {
    resolve_inner(|creds| {
        force_refresh || is_access_token_expiring(&creds.access_token, REFRESH_SKEW_SECONDS)
    })
    .await
}

/// Call after a 401. `failed_access_token` is the token the rejected request
/// used. Under the lock, refresh only if the file token still equals the failed
/// one (or is itself near expiry) — if another in-process agent already
/// refreshed, reuse its token, collapsing N refreshes from a 401 burst into one.
pub async fn resolve_credentials_after_401(
    failed_access_token: &str,
) -> Result<CodexCredentials> {
    resolve_inner(|creds| should_refresh_after_401(&creds.access_token, failed_access_token)).await
}

/// Pure refresh-decision for the 401 path. Refresh if the file token equals the
/// failed one (nobody refreshed) or is near expiry.
fn should_refresh_after_401(current_access: &str, failed_access: &str) -> bool {
    current_access == failed_access
        || is_access_token_expiring(current_access, REFRESH_SKEW_SECONDS)
}

/// Validate that `url` points at a trusted host. Allows only https + the
/// `chatgpt.com` / `openai.com` host suffixes. The user can explicitly opt out
/// of this check for self-hosting / test mocking by setting
/// `CODEX_ALLOW_INSECURE_BASE_URL=1`.
pub fn validate_base_url(url: &str) -> Result<()> {
    if std::env::var("CODEX_ALLOW_INSECURE_BASE_URL")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        return Ok(());
    }
    // Use the URL parser, not manual split: with `https://evil.com#@chatgpt.com`
    // (fragment/query/userinfo evasion), a hand-rolled parser and the real HTTP
    // client can disagree on the host and leak the token. Parse once, reject any
    // suspicious component.
    let parsed = url::Url::parse(url)
        .with_context(|| format!("base_url `{url}` is not a valid URL"))?;
    if parsed.scheme() != "https" {
        bail!(
            "base_url must be https (`{url}`). \
             To allow an insecure URL for testing, set CODEX_ALLOW_INSECURE_BASE_URL=1."
        );
    }
    // userinfo, fragment, query: all evasion vectors and never legitimate in a base_url.
    if !parsed.username().is_empty() || parsed.password().is_some() {
        bail!("base_url `{url}` must not contain userinfo (user:pass@host).");
    }
    if parsed.fragment().is_some() {
        bail!("base_url `{url}` must not contain a fragment (#...).");
    }
    if parsed.query().is_some() {
        bail!("base_url `{url}` must not contain a query string (?...).");
    }
    let host = parsed.host_str().unwrap_or("").to_ascii_lowercase();
    let ok = TRUSTED_HOST_SUFFIXES
        .iter()
        .any(|suf| host == *suf || host.ends_with(&format!(".{suf}")));
    if !ok {
        bail!(
            "host `{host}` of base_url `{url}` is not in the trust list \
             (chatgpt.com, openai.com). \
             To bypass this for self-hosting / testing, set CODEX_ALLOW_INSECURE_BASE_URL=1."
        );
    }
    Ok(())
}

fn insecure_bypass_enabled() -> bool {
    std::env::var("CODEX_ALLOW_INSECURE_BASE_URL")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Whether the host is loopback (127.0.0.0/8, ::1, localhost). Local mock/proxy
/// never leaves the machine, so cleartext to it is allowed.
fn host_is_loopback(parsed: &url::Url) -> bool {
    match parsed.host() {
        Some(url::Host::Ipv4(ip)) => ip.is_loopback(),
        Some(url::Host::Ipv6(ip)) => ip.is_loopback(),
        Some(url::Host::Domain(d)) => d.eq_ignore_ascii_case("localhost"),
        None => false,
    }
}

/// Whether the host is trusted (chatgpt.com/openai.com) or loopback. Used for
/// disk-token destination checks.
fn host_is_trusted_or_loopback(parsed: &url::Url) -> bool {
    if host_is_loopback(parsed) {
        return true;
    }
    match parsed.host() {
        Some(url::Host::Domain(d)) => {
            let h = d.to_ascii_lowercase();
            TRUSTED_HOST_SUFFIXES
                .iter()
                .any(|suf| h == *suf || h.ends_with(&format!(".{suf}")))
        }
        _ => false,
    }
}

/// Validate the token destination. Beyond `validate_base_url`, a disk-loaded
/// token (`from_disk == true`) is sent only to a trusted host (or loopback) even
/// when the `CODEX_ALLOW_INSECURE_BASE_URL` bypass is on: the bypass is a global
/// switch, so a forgotten flag plus `CODEX_BASE_URL=https://evil.com` could leak
/// a real subscription token. The bypass fully applies only to injected
/// (non-disk) credentials; localhost mock/proxy still works.
pub fn validate_token_destination(creds: &CodexCredentials) -> Result<()> {
    // Always pass the base format/trust check (bypass flag respected here).
    validate_base_url(&creds.base_url)?;
    // Extra guard only for disk token + bypass ON.
    if creds.from_disk && insecure_bypass_enabled() {
        guard_disk_token_under_bypass(&creds.base_url)?;
    }
    Ok(())
}

/// Extra guard for a disk token when the bypass is on. Pure (no env reads). Two
/// rules: (1) host must be trusted or loopback; (2) non-loopback hosts require
/// https — the bypass skips `validate_base_url`'s https check entirely, so
/// without this a real token could go cleartext to `http://chatgpt.com`.
fn guard_disk_token_under_bypass(base_url: &str) -> Result<()> {
    let parsed = url::Url::parse(base_url)
        .with_context(|| format!("base_url `{base_url}` is not a valid URL"))?;
    if !host_is_trusted_or_loopback(&parsed) {
        bail!(
            "refusing to send a disk-loaded ChatGPT token to non-trusted host `{}` \
             even though CODEX_ALLOW_INSECURE_BASE_URL is set. The insecure override \
             only fully applies to explicitly injected credentials, not to tokens \
             auto-loaded from ~/.codex/auth.json. Use a loopback/trusted host, or \
             inject credentials directly.",
            parsed.host_str().unwrap_or("")
        );
    }
    if parsed.scheme() != "https" && !host_is_loopback(&parsed) {
        bail!(
            "refusing to send a disk-loaded ChatGPT token over cleartext `{}` \
             (scheme `{}`) even though CODEX_ALLOW_INSECURE_BASE_URL is set. \
             Disk-loaded tokens require https for non-loopback hosts. Use https, \
             a loopback host, or inject credentials directly.",
            base_url,
            parsed.scheme()
        );
    }
    Ok(())
}

// Base URL from CODEX_BASE_URL env, else the default constant.
fn default_base_url() -> String {
    std::env::var("CODEX_BASE_URL")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim_end_matches('/').to_string())
        .unwrap_or_else(|| DEFAULT_BASE_URL.to_string())
}

// Current time as an ISO string for the last_refresh metadata field.
fn now_iso() -> String {
    // A pre-epoch clock reports 0 (1970-01-01); this is only metadata, not fatal.
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let (year, month, day, hour, min, sec) = epoch_to_ymdhms(secs);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{min:02}:{sec:02}Z")
}

/// epoch seconds -> (year, month, day, hour, min, sec). Clamped to
/// `[0, 9999-12-31T23:59:59Z]` — metadata only, so we prioritize panic/overflow
/// safety over precision.
pub(crate) fn epoch_to_ymdhms(epoch: i64) -> (i32, u32, u32, u32, u32, u32) {
    const MAX_EPOCH: i64 = 253_402_300_799; // 9999-12-31T23:59:59Z
    let epoch = epoch.clamp(0, MAX_EPOCH);
    let mut days = epoch.div_euclid(86_400);
    let mut secs_in_day = epoch.rem_euclid(86_400) as u32;
    let hour = secs_in_day / 3600;
    secs_in_day %= 3600;
    let min = secs_in_day / 60;
    let sec = secs_in_day % 60;

    let mut year = 1970i32;
    loop {
        let dy = if is_leap(year) { 366 } else { 365 };
        if days < dy {
            break;
        }
        days -= dy;
        year += 1;
    }
    let months_normal = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let months_leap = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let months = if is_leap(year) { months_leap } else { months_normal };
    let mut month = 1u32;
    for m in months {
        if days < m {
            break;
        }
        days -= m;
        month += 1;
    }
    let day = days as u32 + 1;
    (year, month, day, hour, min, sec)
}

fn is_leap(y: i32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0)
}

// Tests: pure functions only, no network/login.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leap_year_detection() {
        assert!(is_leap(2024));
        assert!(is_leap(2000));
        assert!(!is_leap(2023));
        assert!(!is_leap(1900));
    }

    #[test]
    fn epoch_to_ymdhms_works() {
        assert_eq!(epoch_to_ymdhms(0), (1970, 1, 1, 0, 0, 0));
        assert_eq!(epoch_to_ymdhms(3661), (1970, 1, 1, 1, 1, 1));
        assert_eq!(epoch_to_ymdhms(86_400), (1970, 1, 2, 0, 0, 0));
        assert_eq!(epoch_to_ymdhms(31 * 86_400), (1970, 2, 1, 0, 0, 0));
    }

    #[test]
    fn base_url_only_trusted_hosts() {
        assert!(validate_base_url("https://chatgpt.com/backend-api/codex").is_ok());
        assert!(validate_base_url("https://auth.openai.com").is_ok());
        assert!(validate_base_url("http://chatgpt.com").is_err());
        assert!(validate_base_url("https://evil.com").is_err());
        // Suffix-impersonation domain is rejected.
        assert!(validate_base_url("https://chatgpt.com.evil.com").is_err());
    }

    #[test]
    fn oauth_error_body_parsing() {
        let (code, msg) =
            parse_oauth_error(r#"{"error":{"code":"invalid_grant","message":"nope"}}"#);
        assert_eq!(code.as_deref(), Some("invalid_grant"));
        assert!(msg.unwrap().contains("nope"));

        let (code, _) =
            parse_oauth_error(r#"{"error":"invalid_token","error_description":"expired"}"#);
        assert_eq!(code.as_deref(), Some("invalid_token"));

        // Non-JSON yields no code/message.
        let (code, msg) = parse_oauth_error("this is not JSON");
        assert!(code.is_none() && msg.is_none());
    }

    /// Build a fake JWT; only the middle (payload) segment is meaningful.
    fn make_jwt(payload: &serde_json::Value) -> String {
        let body = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(serde_json::to_vec(payload).unwrap());
        format!("aaa.{body}.sig")
    }

    #[test]
    fn jwt_account_id_extraction() {
        let jwt = make_jwt(&serde_json::json!({
            "https://api.openai.com/auth": { "chatgpt_account_id": "acc_123" }
        }));
        let creds = CodexCredentials {
            access_token: jwt,
            refresh_token: "r".into(),
            base_url: DEFAULT_BASE_URL.into(),
            last_refresh: None,
            from_disk: false,
        };
        assert_eq!(creds.chatgpt_account_id().as_deref(), Some("acc_123"));
    }

    #[test]
    fn token_expiry_detection() {
        let future = make_jwt(&serde_json::json!({ "exp": 9_999_999_999_i64 }));
        assert!(!is_access_token_expiring(&future, 120));

        let past = make_jwt(&serde_json::json!({ "exp": 0 }));
        assert!(is_access_token_expiring(&past, 120));

        // Malformed token is treated as expiring.
        assert!(is_access_token_expiring("not-a-jwt", 120));
    }

    #[test]
    fn refresh_after_401_compare_and_skip() {
        let stale = make_jwt(&serde_json::json!({ "exp": 0 }));
        let fresh = make_jwt(&serde_json::json!({ "exp": 9_999_999_999_i64 }));

        // File token still equals the failed one -> refresh.
        assert!(should_refresh_after_401(&stale, &stale));

        // File token differs and is fresh (already refreshed) -> skip.
        assert!(!should_refresh_after_401(&fresh, &stale));

        // Different but near-expiry file token -> refresh.
        let other_expiring = make_jwt(&serde_json::json!({ "exp": 0 }));
        assert!(should_refresh_after_401(&other_expiring, "completely-different-token"));
    }

    #[test]
    fn codex_home_must_be_absolute() {
        #[cfg(unix)]
        let abs = "/tmp/some/codex/home";
        #[cfg(not(unix))]
        let abs = "C:\\codex\\home";
        let p = codex_home_to_auth_path(abs).unwrap();
        assert!(p.is_absolute());
        assert_eq!(p.file_name().unwrap(), "auth.json");

        // Relative paths are rejected (would drop tokens in cwd).
        assert!(codex_home_to_auth_path(".codex").is_err());
        assert!(codex_home_to_auth_path("relative/dir").is_err());
    }

    #[test]
    fn token_destination_host_classification() {
        let trusted_or_loop = |u: &str| {
            host_is_trusted_or_loopback(&url::Url::parse(u).unwrap())
        };
        assert!(trusted_or_loop("https://chatgpt.com/backend-api/codex"));
        assert!(trusted_or_loop("https://auth.openai.com"));
        assert!(trusted_or_loop("http://127.0.0.1:8080/backend-api/codex"));
        assert!(trusted_or_loop("http://localhost:3000"));
        assert!(trusted_or_loop("http://[::1]:9000"));
        assert!(!trusted_or_loop("https://evil.com"));
        assert!(!trusted_or_loop("https://chatgpt.com.evil.com"));
        assert!(!trusted_or_loop("http://10.0.0.5")); // private, but not loopback
    }

    #[test]
    fn base_url_rejects_evasion_tricks() {
        assert!(validate_base_url("https://evil.com#@chatgpt.com/backend-api/codex").is_err());
        assert!(validate_base_url("https://chatgpt.com@evil.com").is_err());
        assert!(validate_base_url("https://user:pass@chatgpt.com").is_err());
        assert!(validate_base_url("https://evil.com?host=chatgpt.com").is_err());
        assert!(validate_base_url("not a url").is_err());
        assert!(validate_base_url("https://chatgpt.com/backend-api/codex").is_ok());
    }

    #[test]
    fn redacted_debug_hides_tokens() {
        let creds = CodexCredentials {
            access_token: "SUPER_SECRET_ACCESS".into(),
            refresh_token: "SUPER_SECRET_REFRESH".into(),
            base_url: DEFAULT_BASE_URL.into(),
            last_refresh: None,
            from_disk: false,
        };
        let dumped = format!("{creds:?}");
        assert!(!dumped.contains("SUPER_SECRET_ACCESS"), "access token leaked: {dumped}");
        assert!(!dumped.contains("SUPER_SECRET_REFRESH"), "refresh token leaked: {dumped}");
        assert!(dumped.contains("[redacted]"));
        assert!(dumped.contains(DEFAULT_BASE_URL));
    }

    #[test]
    fn device_code_prompt_debug_redacts_user_code() {
        let p = DeviceCodePrompt {
            verification_url: "https://auth.openai.com/codex/device",
            user_code: "SECRET-USER-CODE",
        };
        let dumped = format!("{p:?}");
        assert!(!dumped.contains("SECRET-USER-CODE"), "user_code leaked: {dumped}");
        assert!(dumped.contains("[redacted]"));
        // The (non-secret) verification URL stays visible for debugging.
        assert!(dumped.contains("auth.openai.com/codex/device"));
    }

    #[test]
    fn load_rejects_corrupt_file() {
        use std::io::Write;
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!("codex-oauth-test-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let auth = dir.join("auth.json");
        std::fs::File::create(&auth)
            .unwrap()
            .write_all(b"{ this is not valid json")
            .unwrap();

        // edition 2024: set_var/remove_var are unsafe; nothing else reads CODEX_HOME here.
        unsafe { std::env::set_var("CODEX_HOME", &dir) };
        let result = load_codex_cli_tokens();
        unsafe { std::env::remove_var("CODEX_HOME") };
        let _ = std::fs::remove_dir_all(&dir);

        assert!(result.is_err(), "corrupt token store should be an error");
    }

    #[test]
    fn disk_token_under_bypass_blocks_cleartext() {
        // Regression guard: disk token never goes cleartext, even to a trusted host.
        assert!(
            guard_disk_token_under_bypass("http://chatgpt.com/backend-api/codex").is_err(),
            "cleartext http to a trusted host must be refused for disk tokens"
        );
        assert!(
            guard_disk_token_under_bypass("http://auth.openai.com").is_err(),
            "cleartext http to openai.com must be refused for disk tokens"
        );
        assert!(guard_disk_token_under_bypass("https://chatgpt.com/backend-api/codex").is_ok());
        // loopback allows http for local mock/proxy.
        assert!(guard_disk_token_under_bypass("http://127.0.0.1:8080").is_ok());
        assert!(guard_disk_token_under_bypass("http://localhost:3000").is_ok());
        assert!(guard_disk_token_under_bypass("https://evil.com").is_err());
    }

    #[test]
    fn auth_file_without_tokens_is_not_corrupt() {
        // Regression guard: valid JSON lacking tokens parses as empty (not corrupt).
        let f: AuthFile = serde_json::from_str(r#"{"OPENAI_API_KEY":"sk-test"}"#).unwrap();
        assert!(f.tokens.access_token.is_empty());
        assert!(f.tokens.refresh_token.is_empty());
        assert_eq!(f.openai_api_key.as_deref(), Some("sk-test"));
        let f2: AuthFile = serde_json::from_str(r#"{"tokens":{"access_token":"a"}}"#).unwrap();
        assert_eq!(f2.tokens.access_token, "a");
        assert!(f2.tokens.refresh_token.is_empty());
        // Broken JSON still fails to parse (corrupt).
        assert!(serde_json::from_str::<AuthFile>("{ not valid json").is_err());
    }
}
