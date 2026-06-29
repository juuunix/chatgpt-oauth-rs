//! HTTP client for `chatgpt.com/backend-api/codex/responses`.
//!
//! Async (tokio + reqwest). The ChatGPT backend rejects `stream: false`, so
//! all calls are SSE. `send_message` is a convenience wrapper that drives the
//! stream to completion and returns a typed [`Response`](crate::Response)
//! (raw `Value` still available via `Response::raw()`).

use std::future::Future;
use std::sync::OnceLock;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use futures_util::stream::{Stream, StreamExt};
use reqwest::header::{HeaderMap, HeaderValue};
use serde_json::{Value, json};

use crate::auth::{
    CodexCredentials, MAX_ERROR_BODY, bounded_text, resolve_credentials,
    resolve_credentials_after_401, validate_token_destination,
};
use crate::error::ClientError;

/// SSE accumulator cap: a single event over 16MB is a protocol fault, abort.
const MAX_SSE_BUFFER: usize = 16 * 1024 * 1024;
/// Max `data:` lines per event. Empty `data:` lines barely move the byte cap
/// but keep growing the Vec; cap line count to stop empty-line floods.
const MAX_SSE_DATA_LINES: usize = 65_536;
/// Cap on accumulated output text per response, to bound unending streams.
const MAX_RESPONSE_TEXT_BYTES: usize = 64 * 1024 * 1024;
/// Cap on output_items per response, to bound unending streams.
const MAX_RESPONSE_OUTPUT_ITEMS: usize = 100_000;
/// Max idle between SSE chunks before aborting; keeps a silent backend from hanging.
const SSE_IDLE_TIMEOUT: Duration = Duration::from_secs(120);
const DEFAULT_MAX_RETRIES: u32 = 2;
/// Cap on server `Retry-After` (429); an unbounded value could pin a task for hours.
const MAX_RETRY_AFTER: Duration = Duration::from_secs(60);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// Process-wide shared client: reuses the connection pool and TLS sessions
/// across repeated calls to the same host (building per-call defeats both).
pub(crate) fn shared_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .build()
            // Only fails on unrecoverable env issues (TLS backend init).
            .expect("failed to build shared reqwest client")
    })
}
const BACKOFF_INITIAL_MS: u64 = 200;
const BACKOFF_MAX_MS: u64 = 16_000;

/// Exponential backoff (200, 400, 800ms ... capped at 16s) with ±10% jitter to
/// avoid thundering herd. Uses system-time nanos as entropy (no rand dep).
fn backoff_delay(attempt: u32) -> Duration {
    let exp = BACKOFF_INITIAL_MS.saturating_mul(2u64.saturating_pow(attempt.saturating_sub(1)));
    let base = exp.min(BACKOFF_MAX_MS);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let frac = (nanos % 1000) as f64 / 1000.0; // 0.0..1.0
    let mult = 0.9 + 0.2 * frac; // 0.9..1.1
    Duration::from_millis((base as f64 * mult) as u64)
}

/// Retry an async op with backoff. Server `Retry-After` (429) takes priority
/// over computed backoff.
///
/// `idempotent`: whether retrying can duplicate side effects.
/// - `true`  → GETs (list_models/usage): retry broadly via `is_retryable()`.
/// - `false` → side-effecting POST (`/responses`): only retry errors where the
///   server provably never received the request (connect failure / 429) via
///   `is_retryable_non_idempotent()`, to avoid double tool runs and billing.
async fn with_retry<T, F, Fut>(
    max_retries: u32,
    idempotent: bool,
    mut op: F,
) -> Result<T, ClientError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, ClientError>>,
{
    let mut attempt: u32 = 0;
    loop {
        match op().await {
            Ok(v) => return Ok(v),
            Err(e) => {
                attempt += 1;
                let retryable = if idempotent {
                    e.is_retryable()
                } else {
                    e.is_retryable_non_idempotent()
                };
                if attempt > max_retries || !retryable {
                    if attempt > 1 {
                        tracing::debug!(attempts = attempt, error = %e, "request failed (no more retries)");
                    }
                    return Err(e);
                }
                let delay = e.retry_after().unwrap_or_else(|| backoff_delay(attempt));
                tracing::warn!(
                    attempt,
                    delay_ms = delay.as_millis() as u64,
                    error = %e,
                    "retryable error — backing off and retrying"
                );
                tokio::time::sleep(delay).await;
            }
        }
    }
}

pub(crate) fn build_headers(creds: &CodexCredentials, stream: bool) -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    headers.insert(
        "Authorization",
        HeaderValue::from_str(&format!("Bearer {}", creds.access_token))?,
    );
    headers.insert("Content-Type", HeaderValue::from_static("application/json"));
    headers.insert(
        "Accept",
        if stream {
            HeaderValue::from_static("text/event-stream")
        } else {
            HeaderValue::from_static("application/json")
        },
    );
    headers.insert(
        "User-Agent",
        HeaderValue::from_static("codex_cli_rs/0.0.0 (chatgpt-oauth)"),
    );
    headers.insert("originator", HeaderValue::from_static("codex_cli_rs"));
    // account_id comes from an *unverified* JWT claim — used only as a routing
    // hint. The backend itself authorizes via the access_token.
    if let Some(account_id) = creds.chatgpt_account_id()
        && let Ok(v) = HeaderValue::from_str(&account_id) {
            headers.insert("ChatGPT-Account-ID", v);
        }
    Ok(headers)
}

pub(crate) fn normalize_input(user_message: &str) -> Value {
    json!([
        {
            "role": "user",
            "content": [{"type": "input_text", "text": user_message}]
        }
    ])
}

/// Parse numeric `Retry-After` (seconds), capped at `MAX_RETRY_AFTER`.
fn parse_retry_after(headers: &HeaderMap) -> Option<Duration> {
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(Duration::from_secs)
        .map(|d| d.min(MAX_RETRY_AFTER))
}

/// Convert a non-success response to a `ClientError`. Body bounded to
/// `MAX_ERROR_BODY` (8KB) for diagnostics. 429 → RateLimited, 5xx → Server, else Http.
async fn http_error(resp: reqwest::Response) -> ClientError {
    let status = resp.status();
    let code = status.as_u16();
    let retry_after = parse_retry_after(resp.headers());
    let body = bounded_text(resp, MAX_ERROR_BODY).await;
    if code == 429 {
        ClientError::RateLimited { retry_after }
    } else if status.is_server_error() {
        ClientError::Server { status: code, body }
    } else {
        ClientError::Http { status: code, body }
    }
}

/// One `/models` entry. High-value fields get accessors; the rest via `raw()`
/// (mirroring the full ~38-field schema would just invite drift).
#[derive(Debug, Clone)]
pub struct Model {
    raw: Value,
}

impl Model {
    pub(crate) fn new(raw: Value) -> Model {
        Model { raw }
    }
    /// Model identifier used in requests (`SendOptions.model`).
    pub fn slug(&self) -> Option<&str> {
        self.raw.get("slug").and_then(|v| v.as_str())
    }
    pub fn display_name(&self) -> Option<&str> {
        self.raw.get("display_name").and_then(|v| v.as_str())
    }
    pub fn description(&self) -> Option<&str> {
        self.raw.get("description").and_then(|v| v.as_str())
    }
    pub fn context_window(&self) -> Option<u64> {
        self.raw.get("context_window").and_then(|v| v.as_u64())
    }
    pub fn max_context_window(&self) -> Option<u64> {
        self.raw.get("max_context_window").and_then(|v| v.as_u64())
    }
    pub fn visibility(&self) -> Option<&str> {
        self.raw.get("visibility").and_then(|v| v.as_str())
    }
    /// Escape hatch for remaining fields.
    pub fn raw(&self) -> &Value {
        &self.raw
    }
}

/// `GET /models` — models available to the current account. Retryable errors
/// back off up to DEFAULT_MAX_RETRIES times.
pub async fn list_models() -> Result<Vec<Model>, ClientError> {
    let raw = with_retry(DEFAULT_MAX_RETRIES, true, || async {
        let creds = resolve_credentials(false).await?;
        list_models_once(&creds, true).await
    })
    .await?;
    Ok(raw.into_iter().map(Model::new).collect())
}

/// One call with already-resolved creds. `refresh_on_401`: when true, refresh
/// token from disk and retry once (disk path); when false, surface the error
/// (test-injection path).
async fn list_models_once(
    creds: &CodexCredentials,
    refresh_on_401: bool,
) -> Result<Vec<Value>, ClientError> {
    validate_token_destination(creds)?;
    let url = format!("{}/models?client_version=1.0.0", creds.base_url);
    let resp = shared_client()
        .get(&url)
        .timeout(Duration::from_secs(15))
        .headers(build_headers(creds, false)?)
        .send()
        .await?;
    // On 401 (disk path only) refresh from disk and retry once, using the
    // refreshed credentials' base_url so we never re-send a fresh token to an
    // arbitrary caller URL.
    let resp = if refresh_on_401 && resp.status() == reqwest::StatusCode::UNAUTHORIZED {
        tracing::debug!("HTTP 401 — refreshing credentials, retrying once");
        let refreshed = resolve_credentials_after_401(&creds.access_token).await?;
        validate_token_destination(&refreshed)?;
        let url2 = format!("{}/models?client_version=1.0.0", refreshed.base_url);
        shared_client()
            .get(&url2)
            .timeout(Duration::from_secs(15))
            .headers(build_headers(&refreshed, false)?)
            .send()
            .await?
    } else {
        resp
    };
    if !resp.status().is_success() {
        return Err(http_error(resp).await);
    }
    let v: Value = resp
        .json()
        .await
        .context("failed to parse /models response JSON")?;
    Ok(v.get("models")
        .and_then(|m| m.as_array())
        .cloned()
        .unwrap_or_default())
}

/// Test-only injection seam: uses the given creds as-is (no 401 refresh).
#[cfg(test)]
async fn list_models_with_creds(creds: &CodexCredentials) -> Result<Vec<Value>, ClientError> {
    with_retry(DEFAULT_MAX_RETRIES, true, || list_models_once(creds, false)).await
}

// Usage / rate-limit — `GET /backend-api/wham/usage`. Same info is also in
// `/responses` `x-codex-*` headers, but capturing those before consuming the
// stream is intrusive, so we use this standalone GET endpoint.

/// One rate-limit window (primary=short, secondary=long).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RateWindow {
    /// Percent used in this window (0-100).
    #[serde(default)]
    pub used_percent: f64,
    /// Window length in seconds.
    #[serde(default)]
    pub limit_window_seconds: u64,
    /// Seconds until reset.
    #[serde(default)]
    pub reset_after_seconds: u64,
    /// Reset time (epoch seconds).
    #[serde(default)]
    pub reset_at: i64,
}

/// Rate-limit state.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RateLimit {
    #[serde(default)]
    pub allowed: bool,
    #[serde(default)]
    pub limit_reached: bool,
    pub primary_window: Option<RateWindow>,
    pub secondary_window: Option<RateWindow>,
}

/// Relevant part of `/wham/usage` (plan + rate-limit). Unknown fields ignored.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Usage {
    /// Plan type (e.g. "pro", "plus").
    pub plan_type: Option<String>,
    /// Main rate-limit; None if absent.
    pub rate_limit: Option<RateLimit>,
}

impl Usage {
    /// Higher of primary/secondary used_percent (0-100); None if unknown.
    pub fn max_used_percent(&self) -> Option<f64> {
        let rl = self.rate_limit.as_ref()?;
        let p = rl.primary_window.as_ref().map(|w| w.used_percent);
        let s = rl.secondary_window.as_ref().map(|w| w.used_percent);
        match (p, s) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        }
    }
}

/// codex base_url (`.../backend-api/codex`) → usage URL (`.../backend-api/wham/usage`).
pub(crate) fn usage_url_from_base(base_url: &str) -> String {
    base_url
        .strip_suffix("/codex")
        .map(|prefix| format!("{prefix}/wham/usage"))
        .unwrap_or_else(|| base_url.replace("/backend-api/codex", "/backend-api/wham/usage"))
}

/// Fetch current account usage / rate-limit (for proactive quota monitoring).
pub async fn fetch_usage() -> Result<Usage, ClientError> {
    with_retry(DEFAULT_MAX_RETRIES, true, || async {
        let creds = resolve_credentials(false).await?;
        fetch_usage_once(&creds, true).await
    })
    .await
}

async fn fetch_usage_once(
    creds: &CodexCredentials,
    refresh_on_401: bool,
) -> Result<Usage, ClientError> {
    validate_token_destination(creds)?;
    let url = usage_url_from_base(&creds.base_url);
    let resp = shared_client()
        .get(&url)
        .timeout(Duration::from_secs(15))
        .headers(build_headers(creds, false)?)
        .send()
        .await?;
    let resp = if refresh_on_401 && resp.status() == reqwest::StatusCode::UNAUTHORIZED {
        tracing::debug!("HTTP 401 — refreshing credentials, retrying once");
        let refreshed = resolve_credentials_after_401(&creds.access_token).await?;
        validate_token_destination(&refreshed)?;
        let url2 = usage_url_from_base(&refreshed.base_url);
        shared_client()
            .get(&url2)
            .timeout(Duration::from_secs(15))
            .headers(build_headers(&refreshed, false)?)
            .send()
            .await?
    } else {
        resp
    };
    if !resp.status().is_success() {
        return Err(http_error(resp).await);
    }
    resp.json::<Usage>()
        .await
        .context("failed to parse /wham/usage JSON")
        .map_err(ClientError::from)
}

/// Test-only injection seam: uses the given creds as-is (no 401 refresh).
#[cfg(test)]
async fn fetch_usage_with_creds(creds: &CodexCredentials) -> Result<Usage, ClientError> {
    with_retry(DEFAULT_MAX_RETRIES, true, || fetch_usage_once(creds, false)).await
}

#[derive(Debug, Clone)]
// `#[non_exhaustive]`: control fields keep growing, so external crates build via
// `SendOptions::new(model)`/`default()` + field edits, not struct literals, making
// future field additions non-breaking. (Adopting it is itself a one-time break.)
#[non_exhaustive]
pub struct SendOptions {
    pub model: String,
    pub instructions: String,
    pub reasoning_effort: Option<String>,
    /// Provide a stable key across calls in the same session to maximize
    /// prefix cache routing stickiness. If `None`, the server allocates a
    /// fresh UUID per call (less sticky).
    pub prompt_cache_key: Option<String>,
    /// Tool spec to forward to the backend, e.g.
    /// `[{"type":"web_search"},{"type":"image_generation","quality":"high"}]`.
    /// Empty -> no `tools` field is added to the request.
    pub tools: Vec<Value>,
    /// Extra connection attempts for retryable errors (429/5xx/transient
    /// network). 0 disables retries. Default 2.
    pub max_retries: u32,
    /// Max idle time between chunks once the stream is open; exceeding it aborts
    /// the stream as a stalled backend. Default `SSE_IDLE_TIMEOUT` (120s).
    pub idle_timeout: Duration,

    // Optional Responses API control fields; None means omit (server default).
    // Only fields this backend actually accepts are exposed.
    /// Tool-use mode: `"auto"`/`"none"`/`"required"` string, or a tool-spec object.
    pub tool_choice: Option<Value>,
    /// Allow multiple tool calls in one turn (server default true).
    pub parallel_tool_calls: Option<bool>,
    /// Output text control (`{"verbosity": "low|medium|high", "format": {...}}`).
    pub text: Option<Value>,
}

impl SendOptions {
    /// Start from defaults with only the model set. Recommended entry point
    /// (`#[non_exhaustive]` blocks struct literals for external crates).
    ///
    /// ```ignore
    /// let mut opts = SendOptions::new("gpt-5.3-codex");
    /// opts.reasoning_effort = Some("high".into());
    /// ```
    pub fn new(model: impl Into<String>) -> Self {
        SendOptions { model: model.into(), ..Default::default() }
    }
}

impl Default for SendOptions {
    fn default() -> Self {
        let model = std::env::var("CODEX_DEFAULT_MODEL")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| "gpt-5.3-codex".to_string());
        Self {
            model,
            instructions: "You are a helpful assistant.".to_string(),
            reasoning_effort: None,
            prompt_cache_key: None,
            tools: Vec::new(),
            max_retries: 2,
            idle_timeout: SSE_IDLE_TIMEOUT,
            tool_choice: None,
            parallel_tool_calls: None,
            text: None,
        }
    }
}

/// Open an SSE stream for a single user message.
///
/// For multi-turn conversations build an `input` array yourself and use
/// [`open_stream_with_input`].
pub async fn open_stream(
    user_message: &str,
    opts: &SendOptions,
) -> Result<impl Stream<Item = Result<Value, ClientError>>, ClientError> {
    open_stream_with_input(normalize_input(user_message), opts).await
}

/// Open an SSE stream from a fully-formed `input` array (multi-turn capable).
///
/// `input` matches the OpenAI Responses API shape:
/// ```json
/// [
///   {"role": "user",      "content": [{"type": "input_text",  "text": "..."}]},
///   {"role": "assistant", "content": [{"type": "output_text", "text": "..."}]},
///   {"role": "user",      "content": [{"type": "input_text",  "text": "..."}]}
/// ]
/// ```
///
/// This call is **non-idempotent** (model run, tool calls, billing), so only
/// errors where the server provably never received the request (connect
/// failure, 429) are retried within `opts.max_retries`. Ambiguous errors
/// (timeout, mid-request drop, 5xx) are surfaced, not retried, to avoid
/// double tool runs and double billing. Errors after the stream opens surface
/// as stream items.
pub async fn open_stream_with_input(
    input: Value,
    opts: &SendOptions,
) -> Result<impl Stream<Item = Result<Value, ClientError>>, ClientError> {
    // Clone input each attempt since the body is re-sent on retry. idempotent=false.
    with_retry(opts.max_retries, false, || async {
        let creds = resolve_credentials(false).await?;
        open_stream_with_input_once(input.clone(), opts, &creds, true).await
    })
    .await
}

async fn open_stream_with_input_once(
    input: Value,
    opts: &SendOptions,
    creds: &CodexCredentials,
    refresh_on_401: bool,
) -> Result<impl Stream<Item = Result<Value, ClientError>> + use<>, ClientError> {
    // `use<>`: the returned stream owns resp.bytes_stream() and borrows neither
    // creds nor opts. Without it, Rust 2024 RPIT default capture would tie the
    // stream to the caller's local creds lifetime and fail to compile.
    validate_token_destination(creds)?;

    let body = build_request_body(input, opts);

    // No overall request timeout on a streaming response (would truncate long
    // replies); connect timeout is on the shared client, idle is watched per-chunk.
    let url = format!("{}/responses", creds.base_url);
    let resp = shared_client()
        .post(&url)
        .headers(build_headers(creds, true)?)
        .json(&body)
        .send()
        .await?;

    // On 401 (disk path only) refresh from disk and retry once. The retry URL is
    // rebuilt from the *refreshed* credentials' base_url so a caller-supplied
    // untrusted base_url cannot capture a fresh token on retry.
    let resp = if refresh_on_401 && resp.status() == reqwest::StatusCode::UNAUTHORIZED {
        tracing::debug!("HTTP 401 — refreshing credentials, retrying once");
        let refreshed = resolve_credentials_after_401(&creds.access_token).await?;
        validate_token_destination(&refreshed)?;
        let url2 = format!("{}/responses", refreshed.base_url);
        shared_client()
            .post(&url2)
            .headers(build_headers(&refreshed, true)?)
            .json(&body)
            .send()
            .await?
    } else {
        resp
    };

    if !resp.status().is_success() {
        return Err(http_error(resp).await);
    }

    // Check content-type before parsing: a 200 with an HTML/JSON error body
    // (gateway page, JSON error) is caught here with its body. A *missing*
    // header is OK — the real backend sometimes streams valid SSE without one;
    // only an explicitly non-`text/event-stream` type is rejected as an error body.
    let ct_header = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_ascii_lowercase());
    let mismatched = matches!(&ct_header, Some(ct) if !ct.contains("text/event-stream"));
    if mismatched {
        let ct = ct_header.unwrap_or_default();
        let body = bounded_text(resp, MAX_ERROR_BODY).await;
        return Err(ClientError::Protocol(format!(
            "expected text/event-stream from /responses but got content-type `{ct}`: {body}"
        )));
    }

    Ok(sse_event_stream(resp.bytes_stream(), opts.idle_timeout))
}

pub(crate) fn build_request_body(input: Value, opts: &SendOptions) -> Value {
    let mut body_map = serde_json::Map::new();
    body_map.insert("model".into(), json!(opts.model));
    body_map.insert("instructions".into(), json!(opts.instructions));
    body_map.insert("input".into(), input);
    body_map.insert("store".into(), json!(false));
    body_map.insert("stream".into(), json!(true));
    if let Some(eff) = &opts.reasoning_effort {
        body_map.insert("reasoning".into(), json!({ "effort": eff }));
        // Request encrypted reasoning content only when reasoning is on, so
        // multi-turn can carry prior reasoning forward without server storage.
        body_map.insert(
            "include".into(),
            json!(["reasoning.encrypted_content"]),
        );
    }
    if let Some(key) = &opts.prompt_cache_key {
        body_map.insert("prompt_cache_key".into(), json!(key));
    }
    if !opts.tools.is_empty() {
        body_map.insert("tools".into(), Value::Array(opts.tools.clone()));
    }
    // Optional control fields — forward only those that are set.
    if let Some(tc) = &opts.tool_choice {
        body_map.insert("tool_choice".into(), tc.clone());
    }
    if let Some(p) = opts.parallel_tool_calls {
        body_map.insert("parallel_tool_calls".into(), json!(p));
    }
    if let Some(t) = &opts.text {
        body_map.insert("text".into(), t.clone());
    }
    Value::Object(body_map)
}

/// Byte stream -> SSE event JSON adapter.
///
/// Conformance points:
/// 1. Buffer raw bytes and only decode UTF-8 at line boundaries — never
///    splits a multibyte character across chunks.
/// 2. Multi-line `data:` fields within an event are joined with `\n` and
///    parsed as a single JSON document (per the SSE spec).
/// 3. EOF without a trailing newline still flushes the final event.
/// 4. The accumulator is capped at MAX_SSE_BUFFER; overflow returns Err.
/// 5. Malformed `data:` JSON is surfaced as Err — never silently dropped,
///    so a truncated `response.failed` cannot be lost.
/// 6. Per-chunk idle timeout to avoid hanging on a silent backend.
/// 7. The accumulated `data:` payload for a single event is capped at
///    MAX_SSE_BUFFER AND at MAX_SSE_DATA_LINES lines, so a server that sends
///    endless `data:` lines (even empty ones) without an event boundary cannot
///    grow memory unbounded.
///
/// Stream items are `ClientError` so callers get one error type across the whole
/// operation (connection setup AND streaming), with `status()/retry_after()` intact.
fn sse_event_stream<S>(
    byte_stream: S,
    idle_timeout: Duration,
) -> impl Stream<Item = Result<Value, ClientError>>
where
    S: Stream<Item = reqwest::Result<bytes::Bytes>> + Unpin,
{
    use futures_util::stream;

    struct State<S> {
        stream: S,
        buf: Vec<u8>,
        data_lines: Vec<String>,
        data_bytes: usize, // accumulated data bytes for the current event
        eof: bool,
        finished: bool,
        idle_timeout: Duration,
    }

    let init = State {
        stream: byte_stream,
        buf: Vec::new(),
        data_lines: Vec::new(),
        data_bytes: 0,
        eof: false,
        finished: false,
        idle_timeout,
    };

    stream::unfold(init, |mut st| async move {
        if st.finished {
            return None;
        }

        loop {
            // 1) Try to extract a full line from the buffer.
            if let Some(idx) = st.buf.iter().position(|b| *b == b'\n') {
                let line_bytes: Vec<u8> = st.buf.drain(..=idx).collect();
                let mut end = line_bytes.len().saturating_sub(1); // skip \n
                if end > 0 && line_bytes[end - 1] == b'\r' {
                    end -= 1;
                }
                let line_str = match std::str::from_utf8(&line_bytes[..end]) {
                    Ok(s) => s.to_string(),
                    Err(e) => {
                        st.finished = true;
                        return Some((
                            Err(ClientError::Protocol(format!("SSE line is not valid UTF-8: {e}"))),
                            st,
                        ));
                    }
                };

                // Empty line -> event boundary.
                if line_str.is_empty() {
                    st.data_bytes = 0; // event boundary — reset counter
                    if let Some(ev) = take_event(&mut st.data_lines) {
                        match ev {
                            Ok(Some(v)) => return Some((Ok(v), st)),
                            Ok(None) => {
                                // [DONE] — returning None ends the unfold.
                                return None;
                            }
                            Err(e) => {
                                st.finished = true;
                                return Some((Err(e), st));
                            }
                        }
                    }
                    continue;
                }

                // Comment line — ignore.
                if line_str.starts_with(':') {
                    continue;
                }

                if let Some(rest) = line_str.strip_prefix("data:") {
                    let v = rest.strip_prefix(' ').unwrap_or(rest);
                    // Count lines as well as bytes: empty `data:` lines don't move
                    // the byte cap but still grow the Vec. +1 is the join newline.
                    st.data_bytes = st.data_bytes.saturating_add(v.len().saturating_add(1));
                    if st.data_bytes > MAX_SSE_BUFFER || st.data_lines.len() >= MAX_SSE_DATA_LINES {
                        st.finished = true;
                        return Some((
                            Err(ClientError::Protocol(format!(
                                "SSE event exceeded data cap ({}MB / {} lines) without an event boundary — protocol fault",
                                MAX_SSE_BUFFER / (1024 * 1024),
                                MAX_SSE_DATA_LINES
                            ))),
                            st,
                        ));
                    }
                    st.data_lines.push(v.to_string());
                }
                // Other fields (event:, id:, retry:) are not used here.
                continue;
            }

            // 2) Buffer has no complete line. If EOF, flush whatever's left.
            if st.eof {
                if !st.buf.is_empty() {
                    let line_bytes = std::mem::take(&mut st.buf);
                    let line_str = match std::str::from_utf8(&line_bytes) {
                        Ok(s) => s.trim_end_matches('\r').to_string(),
                        Err(e) => {
                            st.finished = true;
                            return Some((
                                Err(ClientError::Protocol(format!(
                                    "trailing SSE line is not valid UTF-8: {e}"
                                ))),
                                st,
                            ));
                        }
                    };
                    if !line_str.is_empty() && !line_str.starts_with(':')
                        && let Some(rest) = line_str.strip_prefix("data:") {
                            let v = rest.strip_prefix(' ').unwrap_or(rest);
                            st.data_lines.push(v.to_string());
                        }
                }
                st.finished = true;
                st.data_bytes = 0;
                if let Some(ev) = take_event(&mut st.data_lines) {
                    return match ev {
                        Ok(Some(v)) => Some((Ok(v), st)),
                        Ok(None) => None,
                        Err(e) => Some((Err(e), st)),
                    };
                }
                return None;
            }

            // 3) Pull the next chunk, with an idle timeout.
            let next = tokio::time::timeout(st.idle_timeout, st.stream.next()).await;
            match next {
                Err(_elapsed) => {
                    st.finished = true;
                    return Some((
                        Err(ClientError::Protocol(format!(
                            "Codex SSE idle timeout ({}s) — backend stopped sending data",
                            st.idle_timeout.as_secs()
                        ))),
                        st,
                    ));
                }
                Ok(None) => {
                    st.eof = true;
                    continue;
                }
                Ok(Some(Err(e))) => {
                    st.finished = true;
                    // Transport error mid-stream — preserve as Network.
                    return Some((Err(ClientError::Network(e)), st));
                }
                Ok(Some(Ok(chunk))) => {
                    if st.buf.len().saturating_add(chunk.len()) > MAX_SSE_BUFFER {
                        st.finished = true;
                        return Some((
                            Err(ClientError::Protocol(format!(
                                "SSE buffer cap ({}MB) exceeded — protocol fault",
                                MAX_SSE_BUFFER / (1024 * 1024)
                            ))),
                            st,
                        ));
                    }
                    st.buf.extend_from_slice(&chunk);
                }
            }
        }
    })
}

/// Consume the accumulated `data:` lines for the current event.
/// `Ok(Some(v))` — a parsed JSON event. `Ok(None)` — [DONE]. `Err(_)` — parse fault.
fn take_event(data_lines: &mut Vec<String>) -> Option<Result<Option<Value>, ClientError>> {
    if data_lines.is_empty() {
        return None;
    }
    let payload = std::mem::take(data_lines).join("\n");
    let trimmed = payload.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed == "[DONE]" {
        return Some(Ok(None));
    }
    match serde_json::from_str::<Value>(trimmed) {
        Ok(v) => Some(Ok(Some(v))),
        // Withhold the raw payload from the error — it could leak prompt/output
        // fragments into logs. Report length only.
        Err(e) => Some(Err(ClientError::Protocol(format!(
            "failed to parse SSE data payload as JSON: {e} ({} bytes withheld)",
            trimmed.len()
        )))),
    }
}

/// Send a single user message and return the typed [`Response`](crate::Response),
/// combining deltas internally. (Raw Value via `Response::raw()` / `into_raw()`.)
pub async fn send_message(user_message: &str, opts: &SendOptions) -> Result<crate::Response, ClientError> {
    send_with_input(normalize_input(user_message), opts).await
}

/// Multi-turn variant of `send_message`: drive a full `input` array to completion
/// and synthesize a typed [`Response`](crate::Response). For live streaming use
/// [`open_event_stream_with_input`](crate::open_event_stream_with_input).
///
/// This backend is `store:false` — the server keeps no history, so prior turns
/// must be resent in full as [`InputItem`](crate::InputItem)s each call.
pub async fn send_with_input(input: Value, opts: &SendOptions) -> Result<crate::Response, ClientError> {
    let stream = open_stream_with_input(input, opts).await?;
    Ok(crate::Response::new(drive_stream_to_response(stream).await?))
}

/// Test-only injection seam: uses the given creds as-is (no 401 refresh).
#[cfg(test)]
async fn send_message_with_creds(
    user_message: &str,
    opts: &SendOptions,
    creds: &CodexCredentials,
) -> Result<Value, ClientError> {
    let input = normalize_input(user_message);
    let stream = with_retry(opts.max_retries, false, || {
        open_stream_with_input_once(input.clone(), opts, creds, false)
    })
    .await?;
    drive_stream_to_response(stream).await
}

/// Drive an open SSE stream to completion into a final response object,
/// collecting deltas/items and surfacing terminal states (failed/incomplete) as errors.
async fn drive_stream_to_response(
    stream: impl Stream<Item = Result<Value, ClientError>>,
) -> Result<Value, ClientError> {
    let mut stream = Box::pin(stream);
    let mut final_response: Option<Value> = None;
    let mut text_deltas: Vec<String> = Vec::new();
    let mut text_total: usize = 0; // accumulated text_deltas bytes
    let mut output_items: Vec<Value> = Vec::new();

    while let Some(ev) = stream.next().await {
        let ev = ev?;
        let et = ev.get("type").and_then(|t| t.as_str()).unwrap_or("");
        match et {
            "response.completed" => {
                if let Some(r) = ev.get("response").cloned() {
                    final_response = Some(r);
                }
            }
            "response.failed" => {
                let err = ev
                    .get("response")
                    .and_then(|r| r.get("error"))
                    .or_else(|| ev.get("error"))
                    .cloned()
                    .unwrap_or(Value::Null);
                return Err(ClientError::Protocol(format!("Codex response failed: {err}")));
            }
            // Terminal incomplete event — catch so it isn't synthesized as success.
            "response.incomplete" => {
                let detail = ev
                    .get("response")
                    .and_then(|r| r.get("incomplete_details"))
                    .or_else(|| ev.get("incomplete_details"))
                    .cloned()
                    .unwrap_or(Value::Null);
                return Err(ClientError::Protocol(format!(
                    "Codex response incomplete: {detail}"
                )));
            }
            "response.output_text.delta" => {
                if let Some(delta) = ev.get("delta").and_then(|d| d.as_str()) {
                    // Cap accumulated text to bound an unending stream.
                    text_total = text_total.saturating_add(delta.len());
                    if text_total > MAX_RESPONSE_TEXT_BYTES {
                        return Err(ClientError::Protocol(format!(
                            "Codex response exceeded max accumulated text ({}MB) — aborting",
                            MAX_RESPONSE_TEXT_BYTES / (1024 * 1024)
                        )));
                    }
                    text_deltas.push(delta.to_string());
                }
            }
            "response.output_item.done" => {
                if let Some(item) = ev.get("item").cloned() {
                    // Cap output_item count to bound an unending stream.
                    if output_items.len() >= MAX_RESPONSE_OUTPUT_ITEMS {
                        return Err(ClientError::Protocol(format!(
                            "Codex response exceeded max output items ({}) — aborting",
                            MAX_RESPONSE_OUTPUT_ITEMS
                        )));
                    }
                    output_items.push(item);
                }
            }
            _ => {}
        }
    }

    // If `response.completed` never arrived but we did collect deltas/items,
    // synthesize a response (tolerates event-name changes or omissions).
    let mut response = match final_response {
        Some(r) => r,
        None => {
            if text_deltas.is_empty() && output_items.is_empty() {
                return Err(ClientError::Protocol(
                    "Codex stream ended without response.completed or any deltas".into(),
                ));
            }
            json!({ "output": [] })
        }
    };

    // Refuse anything that's not an object — would otherwise panic on indexing.
    if !response.is_object() {
        return Err(ClientError::Protocol(format!(
            "Codex response is not an object: {response}"
        )));
    }

    // Terminal-status check: even with a `response.completed` event, the response
    // object's status may be failed/incomplete or carry an error. Returning it as
    // success would silently hand the caller an empty/error response.
    if let Some(status) = response.get("status").and_then(|s| s.as_str())
        && matches!(status, "failed" | "cancelled" | "incomplete" | "expired")
    {
        let detail = response
            .get("error")
            .filter(|e| !e.is_null())
            .or_else(|| response.get("incomplete_details"))
            .cloned()
            .unwrap_or(Value::Null);
        return Err(ClientError::Protocol(format!(
            "Codex response terminal status `{status}`: {detail}"
        )));
    }
    // Treat a non-null error as failure even without a status.
    if let Some(err) = response.get("error")
        && !err.is_null()
    {
        return Err(ClientError::Protocol(format!(
            "Codex response carried an error: {err}"
        )));
    }

    let output_empty = response
        .get("output")
        .and_then(|o| o.as_array())
        .is_none_or(|a| a.is_empty());
    if output_empty {
        let obj = response.as_object_mut().expect("checked is_object above");
        if !output_items.is_empty() {
            obj.insert("output".into(), Value::Array(output_items));
        } else if !text_deltas.is_empty() {
            obj.insert(
                "output".into(),
                json!([
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text_deltas.join("")}]
                    }
                ]),
            );
        }
    }
    Ok(response)
}

/// Pull the concatenated assistant text out of a response dict.
pub fn extract_text(response: &Value) -> String {
    let Some(items) = response.get("output").and_then(|o| o.as_array()) else {
        return String::new();
    };
    let mut out = String::new();
    for item in items {
        if item.get("type").and_then(|t| t.as_str()) != Some("message") {
            continue;
        }
        let Some(contents) = item.get("content").and_then(|c| c.as_array()) else {
            continue;
        };
        for c in contents {
            if c.get("type").and_then(|t| t.as_str()) == Some("output_text")
                && let Some(t) = c.get("text").and_then(|t| t.as_str()) {
                    out.push_str(t);
                }
        }
    }
    out
}

// Tests — pure functions + SSE parser. No network (fake byte streams injected).
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_input_shape() {
        let v = normalize_input("hi");
        let first = &v[0];
        assert_eq!(first["role"], "user");
        assert_eq!(first["content"][0]["type"], "input_text");
        assert_eq!(first["content"][0]["text"], "hi");
    }

    #[test]
    fn retry_after_is_capped() {
        use reqwest::header::{HeaderValue, RETRY_AFTER};
        let mut huge = HeaderMap::new();
        huge.insert(RETRY_AFTER, HeaderValue::from_static("999999"));
        assert_eq!(parse_retry_after(&huge), Some(MAX_RETRY_AFTER));
        let mut small = HeaderMap::new();
        small.insert(RETRY_AFTER, HeaderValue::from_static("5"));
        assert_eq!(parse_retry_after(&small), Some(Duration::from_secs(5)));
        assert_eq!(parse_retry_after(&HeaderMap::new()), None);
        let mut date = HeaderMap::new();
        date.insert(RETRY_AFTER, HeaderValue::from_static("Wed, 21 Oct 2015 07:28:00 GMT"));
        assert_eq!(parse_retry_after(&date), None);
    }

    #[test]
    fn build_request_body_required_fields() {
        let opts = SendOptions {
            model: "gpt-5.3-codex".into(),
            instructions: "sys".into(),
            ..SendOptions::default()
        };
        let body = build_request_body(normalize_input("hi"), &opts);
        assert_eq!(body["model"], "gpt-5.3-codex");
        assert_eq!(body["instructions"], "sys");
        assert_eq!(body["stream"], true);
        assert_eq!(body["store"], false);
        assert!(body.get("input").is_some());
        for k in [
            "reasoning", "include", "prompt_cache_key", "tools",
            "tool_choice", "parallel_tool_calls", "text",
        ] {
            assert!(body.get(k).is_none(), "{k} should be absent");
        }
    }

    #[test]
    fn build_request_body_optional_control_fields() {
        let opts = SendOptions {
            tool_choice: Some(json!("required")),
            parallel_tool_calls: Some(false),
            text: Some(json!({ "verbosity": "low" })),
            ..SendOptions::default()
        };
        let body = build_request_body(normalize_input("hi"), &opts);
        assert_eq!(body["tool_choice"], "required");
        assert_eq!(body["parallel_tool_calls"], false);
        assert_eq!(body["text"]["verbosity"], "low");
    }

    #[test]
    fn build_request_body_adds_fields_when_set() {
        let opts = SendOptions {
            reasoning_effort: Some("high".into()),
            prompt_cache_key: Some("k1".into()),
            tools: vec![json!({"type": "web_search"})],
            ..SendOptions::default()
        };
        let body = build_request_body(normalize_input("hi"), &opts);
        assert_eq!(body["reasoning"]["effort"], "high");
        assert_eq!(body["include"][0], "reasoning.encrypted_content");
        assert_eq!(body["prompt_cache_key"], "k1");
        assert_eq!(body["tools"][0]["type"], "web_search");
    }

    #[test]
    fn extract_text_joins_message_output_text() {
        let resp = json!({
            "output": [
                { "type": "reasoning", "content": [{"type": "output_text", "text": "ignored"}] },
                { "type": "message", "role": "assistant",
                  "content": [
                      {"type": "output_text", "text": "Hello, "},
                      {"type": "output_text", "text": "world"}
                  ]
                }
            ]
        });
        assert_eq!(extract_text(&resp), "Hello, world");
    }

    #[test]
    fn extract_text_empty_when_no_output() {
        assert_eq!(extract_text(&json!({})), "");
    }

    #[test]
    fn model_typed_accessors() {
        let m = Model::new(json!({
            "slug":"gpt-5.3-codex","display_name":"GPT-5.3 Codex",
            "context_window":272000,"max_context_window":400000,"visibility":"public","extra":1
        }));
        assert_eq!(m.slug(), Some("gpt-5.3-codex"));
        assert_eq!(m.display_name(), Some("GPT-5.3 Codex"));
        assert_eq!(m.context_window(), Some(272000));
        assert_eq!(m.max_context_window(), Some(400000));
        assert_eq!(m.visibility(), Some("public"));
        assert_eq!(m.raw()["extra"], 1);
        assert_eq!(Model::new(json!({})).slug(), None);
    }

    /// Feed fake byte chunks through sse_event_stream and collect the events.
    async fn run_sse(chunks: Vec<bytes::Bytes>) -> Vec<Result<Value, ClientError>> {
        use futures_util::stream;
        let items: Vec<reqwest::Result<bytes::Bytes>> = chunks.into_iter().map(Ok).collect();
        let byte_stream = stream::iter(items);
        let mut s = Box::pin(sse_event_stream(byte_stream, Duration::from_secs(5)));
        let mut out = Vec::new();
        while let Some(ev) = s.next().await {
            out.push(ev);
        }
        out
    }

    #[tokio::test]
    async fn sse_single_event() {
        let events = run_sse(vec![bytes::Bytes::from("data: {\"type\":\"x\",\"n\":1}\n\n")]).await;
        assert_eq!(events.len(), 1);
        let v = events[0].as_ref().unwrap();
        assert_eq!(v["type"], "x");
        assert_eq!(v["n"], 1);
    }

    #[tokio::test]
    async fn sse_done_ends_stream() {
        let events = run_sse(vec![bytes::Bytes::from(
            "data: {\"type\":\"a\"}\n\ndata: [DONE]\n\ndata: {\"type\":\"b\"}\n\n",
        )])
        .await;
        assert_eq!(events.len(), 1); // b is after [DONE], so excluded
        assert_eq!(events[0].as_ref().unwrap()["type"], "a");
    }

    #[tokio::test]
    async fn sse_multiline_data_joined() {
        let events = run_sse(vec![bytes::Bytes::from("data: {\"type\":\ndata: \"x\"}\n\n")]).await;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].as_ref().unwrap()["type"], "x");
    }

    #[tokio::test]
    async fn sse_chunk_split_multibyte_safe() {
        // Split the 3-byte '한' across two chunks to test multibyte-safe decoding.
        let full = "data: {\"k\":\"한\"}\n\n".as_bytes().to_vec();
        let mid = 13; // mid-way through the '한' bytes
        let c1 = bytes::Bytes::copy_from_slice(&full[..mid]);
        let c2 = bytes::Bytes::copy_from_slice(&full[mid..]);
        let events = run_sse(vec![c1, c2]).await;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].as_ref().unwrap()["k"], "한");
    }

    #[tokio::test]
    async fn sse_malformed_json_surfaces_error() {
        let events = run_sse(vec![bytes::Bytes::from("data: {not json}\n\n")]).await;
        assert_eq!(events.len(), 1);
        assert!(events[0].is_err());
    }

    #[tokio::test]
    async fn sse_empty_data_line_flood_is_capped() {
        // Empty `data:` lines without an event boundary must hit the line cap.
        let flood = "data:\n".repeat(MAX_SSE_DATA_LINES + 10);
        let events = run_sse(vec![bytes::Bytes::from(flood)]).await;
        let last = events.last().expect("should yield at least the cap error");
        assert!(last.is_err(), "empty-data-line flood must surface an error");
        let msg = last.as_ref().unwrap_err().to_string();
        assert!(
            msg.contains("lines") || msg.contains("data cap"),
            "expected data-cap error, got: {msg}"
        );
    }

    #[tokio::test]
    async fn sse_flush_on_eof_without_newline() {
        let events = run_sse(vec![bytes::Bytes::from("data: {\"type\":\"last\"}")]).await;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].as_ref().unwrap()["type"], "last");
    }

    #[test]
    fn backoff_exponential_capped() {
        let in_range = |d: Duration, base_ms: u64| {
            let ms = d.as_millis() as u64;
            ms >= (base_ms as f64 * 0.9) as u64 && ms <= (base_ms as f64 * 1.1) as u64
        };
        assert!(in_range(backoff_delay(1), 200));
        assert!(in_range(backoff_delay(2), 400));
        assert!(in_range(backoff_delay(3), 800));
        assert!(backoff_delay(30).as_millis() as u64 <= (BACKOFF_MAX_MS as f64 * 1.1) as u64);
    }

    #[tokio::test]
    async fn with_retry_retries_then_succeeds() {
        use std::cell::Cell;
        let calls = Cell::new(0u32);
        let r: Result<u8, ClientError> = with_retry(5, true, || {
            let n = calls.get() + 1;
            calls.set(n);
            async move {
                if n < 3 {
                    Err(ClientError::RateLimited { retry_after: Some(Duration::from_millis(1)) })
                } else {
                    Ok(42u8)
                }
            }
        })
        .await;
        assert_eq!(r.unwrap(), 42);
        assert_eq!(calls.get(), 3); // 2 failures + 1 success
    }

    #[tokio::test]
    async fn with_retry_non_retryable_immediate() {
        use std::cell::Cell;
        let calls = Cell::new(0u32);
        let r: Result<u8, ClientError> = with_retry(5, true, || {
            calls.set(calls.get() + 1);
            async { Err(ClientError::Http { status: 400, body: "bad".into() }) }
        })
        .await;
        assert!(r.is_err());
        assert_eq!(calls.get(), 1); // 400 is not retried
    }

    #[tokio::test]
    async fn with_retry_non_idempotent_skips_ambiguous_errors() {
        use std::cell::Cell;
        // Non-idempotent: 5xx may have been processed, so don't retry.
        let calls = Cell::new(0u32);
        let r: Result<u8, ClientError> = with_retry(5, false, || {
            calls.set(calls.get() + 1);
            async { Err(ClientError::Server { status: 503, body: "x".into() }) }
        })
        .await;
        assert!(r.is_err());
        assert_eq!(calls.get(), 1, "5xx must NOT be retried for non-idempotent ops");

        // 429 is safe to retry even non-idempotent (server refused the request).
        let calls2 = Cell::new(0u32);
        let r2: Result<u8, ClientError> = with_retry(3, false, || {
            let n = calls2.get() + 1;
            calls2.set(n);
            async move {
                if n < 2 {
                    Err(ClientError::RateLimited { retry_after: Some(Duration::from_millis(1)) })
                } else {
                    Ok(7u8)
                }
            }
        })
        .await;
        assert_eq!(r2.unwrap(), 7);
        assert_eq!(calls2.get(), 2, "429 should still be retried for non-idempotent ops");
    }

    #[test]
    fn usage_json_parsing() {
        let raw = r#"{
            "plan_type": "pro",
            "rate_limit": {
                "allowed": true,
                "limit_reached": false,
                "primary_window": {
                    "used_percent": 12.5,
                    "limit_window_seconds": 18000,
                    "reset_after_seconds": 1700,
                    "reset_at": 1780034186
                },
                "secondary_window": {
                    "used_percent": 40.0,
                    "limit_window_seconds": 604800,
                    "reset_after_seconds": 357129,
                    "reset_at": 1780389606
                }
            },
            "credits": { "balance": "0" }
        }"#;
        let u: Usage = serde_json::from_str(raw).unwrap();
        assert_eq!(u.plan_type.as_deref(), Some("pro"));
        let rl = u.rate_limit.as_ref().unwrap();
        assert!(rl.allowed && !rl.limit_reached);
        assert_eq!(rl.primary_window.as_ref().unwrap().used_percent, 12.5);
        assert_eq!(rl.secondary_window.as_ref().unwrap().reset_at, 1780389606);
        assert_eq!(u.max_used_percent(), Some(40.0));
    }

    #[test]
    fn usage_url_derivation() {
        assert_eq!(
            usage_url_from_base("https://chatgpt.com/backend-api/codex"),
            "https://chatgpt.com/backend-api/wham/usage"
        );
    }

    #[test]
    fn usage_parses_without_rate_limit() {
        let u: Usage = serde_json::from_str(r#"{"plan_type":"plus"}"#).unwrap();
        assert_eq!(u.plan_type.as_deref(), Some("plus"));
        assert!(u.rate_limit.is_none());
        assert_eq!(u.max_used_percent(), None);
    }

    #[tokio::test]
    async fn with_retry_returns_last_error_when_exhausted() {
        use std::cell::Cell;
        let calls = Cell::new(0u32);
        let r: Result<u8, ClientError> = with_retry(2, true, || {
            calls.set(calls.get() + 1);
            async { Err(ClientError::Server { status: 503, body: "x".into() }) }
        })
        .await;
        assert!(matches!(r, Err(ClientError::Server { status: 503, .. })));
        assert_eq!(calls.get(), 3); // 1 initial + 2 retries
    }
}

// Integration tests — verify network paths against a wiremock HTTP server.
// The disk path has no public base_url injection seam, so these use the
// test-only `*_with_creds` seam to point fake creds at the mock server.
#[cfg(test)]
mod http_tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// Fake creds pointing at the mock server, with a realistic
    /// `/backend-api/codex` base_url so URL derivation (e.g. /wham/usage) works.
    fn fake_creds(server_uri: &str) -> CodexCredentials {
        // Disable base-url trust check to allow http://127.0.0.1.
        unsafe {
            std::env::set_var("CODEX_ALLOW_INSECURE_BASE_URL", "1");
        }
        CodexCredentials {
            access_token: "test-access".into(),
            refresh_token: "test-refresh".into(),
            base_url: format!("{server_uri}/backend-api/codex"),
            last_refresh: None,
            from_disk: false, // test-injected token
        }
    }

    #[tokio::test]
    async fn list_models_success_parsing() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/backend-api/codex/models"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(json!({"models":[{"slug":"gpt-5.3-codex"}]})),
            )
            .mount(&server)
            .await;

        let c = fake_creds(&server.uri());
        let models = list_models_with_creds(&c).await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0]["slug"], "gpt-5.3-codex");
    }

    #[tokio::test]
    async fn list_models_429_retry_then_success() {
        let server = MockServer::start().await;
        // First call 429, then 200.
        Mock::given(method("GET"))
            .and(path("/backend-api/codex/models"))
            .respond_with(ResponseTemplate::new(429))
            .up_to_n_times(1)
            .with_priority(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/backend-api/codex/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"models":[{"slug":"x"}]})))
            .mount(&server)
            .await;

        let c = fake_creds(&server.uri());
        let models = list_models_with_creds(&c).await.unwrap();
        assert_eq!(models.len(), 1); // succeeds after one 429 + retry
    }

    #[tokio::test]
    async fn list_models_500_retry_then_success() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/backend-api/codex/models"))
            .respond_with(ResponseTemplate::new(500))
            .up_to_n_times(1)
            .with_priority(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/backend-api/codex/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"models":[]})))
            .mount(&server)
            .await;

        let c = fake_creds(&server.uri());
        assert!(list_models_with_creds(&c).await.is_ok());
    }

    #[tokio::test]
    async fn list_models_400_no_retry() {
        let server = MockServer::start().await;
        // expect(1): must be called exactly once (no retry).
        Mock::given(method("GET"))
            .and(path("/backend-api/codex/models"))
            .respond_with(ResponseTemplate::new(400).set_body_string("bad request"))
            .expect(1)
            .mount(&server)
            .await;

        let c = fake_creds(&server.uri());
        let err = list_models_with_creds(&c).await.unwrap_err();
        assert_eq!(err.status(), Some(400));
        assert!(!err.is_retryable());
    }

    #[tokio::test]
    async fn send_message_sse_parsing() {
        let server = MockServer::start().await;
        let sse = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello\"}\n\n\
data: {\"type\":\"response.completed\",\"response\":{\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hello\"}]}]}}\n\n\
data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/backend-api/codex/responses"))
            .respond_with(
                // set_body_raw sets content-type explicitly; set_body_string would
                // force text/plain and fail the SSE content-type check.
                ResponseTemplate::new(200).set_body_raw(sse.as_bytes(), "text/event-stream"),
            )
            .mount(&server)
            .await;

        let c = fake_creds(&server.uri());
        let resp = send_message_with_creds("hi", &SendOptions::default(), &c)
            .await
            .unwrap();
        assert_eq!(extract_text(&resp), "Hello");
    }

    #[tokio::test]
    async fn fetch_usage_parsing() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/backend-api/wham/usage"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "plan_type": "pro",
                "rate_limit": {
                    "allowed": true,
                    "limit_reached": false,
                    "primary_window": {"used_percent": 25.0, "limit_window_seconds": 18000, "reset_after_seconds": 1000, "reset_at": 123},
                    "secondary_window": {"used_percent": 5.0, "limit_window_seconds": 604800, "reset_after_seconds": 2000, "reset_at": 456}
                }
            })))
            .mount(&server)
            .await;

        let c = fake_creds(&server.uri());
        let usage = fetch_usage_with_creds(&c).await.unwrap();
        assert_eq!(usage.plan_type.as_deref(), Some("pro"));
        assert_eq!(usage.max_used_percent(), Some(25.0)); // primary(25) > secondary(5)
    }

    #[tokio::test]
    async fn send_message_wrong_content_type_errors() {
        // 200 but a JSON error body instead of SSE — must be caught by the content-type check.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/backend-api/codex/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(r#"{"error":"gateway exploded"}"#.as_bytes(), "application/json"),
            )
            .mount(&server)
            .await;

        let c = fake_creds(&server.uri());
        let err = send_message_with_creds("hi", &SendOptions::default(), &c)
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("text/event-stream"),
            "expected content-type error, got: {msg}"
        );
    }

    #[tokio::test]
    async fn send_message_accepts_missing_content_type() {
        // Regression: the real backend may stream valid SSE with no Content-Type
        // header; missing must not be rejected. set_body_bytes adds no header.
        let server = MockServer::start().await;
        let sse = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello\"}\n\n\
data: {\"type\":\"response.completed\",\"response\":{\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hello\"}]}]}}\n\n\
data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/backend-api/codex/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(sse.as_bytes()))
            .mount(&server)
            .await;

        let c = fake_creds(&server.uri());
        let resp = send_message_with_creds("hi", &SendOptions::default(), &c)
            .await
            .expect("missing content-type with valid SSE body must be accepted");
        assert_eq!(extract_text(&resp), "Hello");
    }

    #[tokio::test]
    async fn send_message_completed_but_failed_status_errors() {
        // response.completed but inner status=failed + error must not leak as success.
        let server = MockServer::start().await;
        let sse = "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"failed\",\"error\":{\"message\":\"model exploded\"},\"output\":[]}}\n\n\
data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/backend-api/codex/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(sse.as_bytes(), "text/event-stream"))
            .mount(&server)
            .await;

        let c = fake_creds(&server.uri());
        let err = send_message_with_creds("hi", &SendOptions::default(), &c)
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("terminal status") && msg.contains("failed"),
            "expected terminal-status error, got: {msg}"
        );
    }

    #[tokio::test]
    async fn send_message_response_incomplete_errors() {
        // Terminal response.incomplete event must surface as an error.
        let server = MockServer::start().await;
        let sse = "data: {\"type\":\"response.incomplete\",\"response\":{\"incomplete_details\":{\"reason\":\"max_output_tokens\"}}}\n\n\
data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/backend-api/codex/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(sse.as_bytes(), "text/event-stream"))
            .mount(&server)
            .await;

        let c = fake_creds(&server.uri());
        let err = send_message_with_creds("hi", &SendOptions::default(), &c)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("incomplete"));
    }
}
