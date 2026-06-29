//! Diagnostic raw-response capture (only compiled under feature = "capture").
//!
//! Captures endpoint responses pre-parse so schema changes can be inspected by
//! eye, unlike the normal typed functions that error at the parse step. Reuses
//! the real internal URL/header/client helpers to avoid drift.

use std::time::Duration;

use anyhow::{Context, Result};
use serde_json::Value;

use crate::auth::{self, resolve_credentials};
use crate::client;

/// Pre-parse HTTP response snapshot.
#[derive(Debug, serde::Serialize)]
pub struct RawCapture {
    pub method: String,
    pub endpoint: String,
    pub status: u16,
    pub body: String,
    /// Best-effort parse (None if not JSON).
    pub body_json: Option<Value>,
}

/// Today's UTC date "YYYY-MM-DD", for capture file/dir names.
pub fn today_utc() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let (y, m, d, _, _, _) = crate::auth::epoch_to_ymdhms(secs);
    format!("{y:04}-{m:02}-{d:02}")
}

async fn snapshot(method: &str, url: &str, resp: reqwest::Response) -> Result<RawCapture> {
    let status = resp.status().as_u16();
    let body = resp.text().await.context("failed to read response body")?;
    let body_json = serde_json::from_str::<Value>(&body).ok();
    Ok(RawCapture {
        method: method.to_string(),
        endpoint: url.to_string(),
        status,
        body,
        body_json,
    })
}

/// `GET /wham/usage` raw, unparsed. Idempotent, safe to repeat; needs a token.
pub async fn usage_raw() -> Result<RawCapture> {
    let creds = resolve_credentials(false).await?;
    // Validate token destination before attaching Authorization (prevents token
    // leak to a malicious base_url under CODEX_ALLOW_INSECURE_BASE_URL).
    auth::validate_token_destination(&creds)?;
    let url = client::usage_url_from_base(&creds.base_url);
    let resp = client::shared_client()
        .get(&url)
        .timeout(Duration::from_secs(15))
        .headers(client::build_headers(&creds, false)?)
        .send()
        .await
        .context("usage request failed")?;
    snapshot("GET", &url, resp).await
}

/// `GET /models` raw, unparsed. Idempotent, safe.
pub async fn models_raw() -> Result<RawCapture> {
    let creds = resolve_credentials(false).await?;
    // Validate token destination before attaching Authorization (see usage_raw).
    auth::validate_token_destination(&creds)?;
    let url = format!("{}/models?client_version=1.0.0", creds.base_url);
    let resp = client::shared_client()
        .get(&url)
        .timeout(Duration::from_secs(15))
        .headers(client::build_headers(&creds, false)?)
        .send()
        .await
        .context("models request failed")?;
    snapshot("GET", &url, resp).await
}

/// device-code POST (`/deviceauth/usercode`) raw. Safe: only starts login (no
/// poll/complete), so no side effects; needs no token.
pub async fn device_usercode_raw() -> Result<RawCapture> {
    let url = auth::DEVICE_USERCODE_URL;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .context("failed to build http client")?;
    let resp = client
        .post(url)
        .json(&serde_json::json!({ "client_id": auth::CLIENT_ID }))
        .send()
        .await
        .context("device usercode request failed")?;
    snapshot("POST", url, resp).await
}

/// `POST /responses` — collect SSE events as a raw, unaggregated Value array.
/// Consumes tokens/quota but is otherwise safe.
pub async fn responses_raw(prompt: &str, opts: &crate::SendOptions) -> Result<Vec<Value>> {
    responses_with_input_raw(crate::client::normalize_input(prompt), opts).await
}

/// `POST /responses` with a full input array (multiturn / tool feedback capture).
pub async fn responses_with_input_raw(
    input: Value,
    opts: &crate::SendOptions,
) -> Result<Vec<Value>> {
    use futures_util::StreamExt;
    let stream = client::open_stream_with_input(input, opts)
        .await
        .context("failed to open /responses stream")?;
    let mut stream = Box::pin(stream);
    let mut events = Vec::new();
    while let Some(ev) = stream.next().await {
        events.push(ev.context("stream event error")?);
    }
    Ok(events)
}

/// Probe whether the backend accepts given built-in tools: sends a minimal
/// `/responses` request and records 200 (accepted, SSE body not read) vs non-200
/// (error body kept up to 8KB). Used to discover the built-in tool catalog.
pub async fn responses_probe_raw(tools: Vec<Value>, prompt: &str) -> Result<RawCapture> {
    let creds = resolve_credentials(false).await?;
    // Validate token destination before attaching Authorization (see usage_raw).
    auth::validate_token_destination(&creds)?;
    let opts = crate::SendOptions { tools, ..Default::default() };
    let body = client::build_request_body(client::normalize_input(prompt), &opts);
    let url = format!("{}/responses", creds.base_url);
    let resp = client::shared_client()
        .post(&url)
        .timeout(Duration::from_secs(30))
        .headers(client::build_headers(&creds, true)?)
        .json(&body)
        .send()
        .await
        .context("responses probe request failed")?;
    let status = resp.status().as_u16();
    let (body, body_json) = if status == 200 {
        ("(accepted — 200, SSE body not captured)".to_string(), None)
    } else {
        let b = crate::auth::bounded_text(resp, 8192).await;
        let j = serde_json::from_str::<Value>(&b).ok();
        (b, j)
    };
    Ok(RawCapture { method: "POST".into(), endpoint: url, status, body, body_json })
}
