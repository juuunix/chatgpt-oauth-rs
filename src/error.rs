//! Client error type — classifies HTTP failures so consumers can build retry
//! strategies programmatically (status preserved as data + `is_retryable*`).

use std::time::Duration;

#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    /// 429. `retry_after` parsed from the `Retry-After` header.
    #[error("rate limited (HTTP 429)")]
    RateLimited { retry_after: Option<Duration> },

    /// 5xx — server-side, usually retryable.
    #[error("server error: HTTP {status}: {body}")]
    Server { status: u16, body: String },

    /// Other non-success (mostly 4xx) — usually not retryable.
    #[error("HTTP {status}: {body}")]
    Http { status: u16, body: String },

    #[error("network error: {0}")]
    Network(#[from] reqwest::Error),

    /// Response violated protocol expectations (SSE parse failure, missing field).
    #[error("protocol error: {0}")]
    Protocol(String),

    /// May wrap an `AuthError`; consumers can `.downcast_ref::<AuthError>()`.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl ClientError {
    /// Retryable for **idempotent** requests (GET-like): retry has no side effects.
    pub fn is_retryable(&self) -> bool {
        match self {
            ClientError::RateLimited { .. } => true,
            ClientError::Server { status, .. } => (500..600).contains(status),
            ClientError::Network(e) => e.is_timeout() || e.is_connect() || e.is_request(),
            _ => false,
        }
    }

    /// Retryable for **non-idempotent** requests (side-effecting POST like
    /// `/responses`): only errors with no duplicate-execution risk.
    /// Criterion: could the server have already processed the request?
    /// - RateLimited (429): server explicitly refused → safe.
    /// - Network(is_connect): failed at connect → server never received it → safe.
    /// - timeout / is_request / 5xx: server may have processed it → don't retry
    ///   (risk of double tool execution / double billing).
    pub fn is_retryable_non_idempotent(&self) -> bool {
        match self {
            ClientError::RateLimited { .. } => true,
            ClientError::Network(e) => e.is_connect(),
            _ => false,
        }
    }

    /// Whether this signals **relogin required** (an `AuthError::ReloginRequired`
    /// wrapped in `Other`), detectable in one branch without downcasting.
    ///
    /// Caveat: if token refresh succeeded but the retried request still 401s, that
    /// surfaces as `Http { status: 401 }` and this returns `false` (relogin may not
    /// be the fix). To also handle that, check `self.status() == Some(401)`.
    pub fn is_relogin_required(&self) -> bool {
        matches!(self, ClientError::Other(e) if crate::auth::is_relogin_required(e))
    }

    /// Server-advertised retry delay, if any.
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            ClientError::RateLimited { retry_after } => *retry_after,
            _ => None,
        }
    }

    /// HTTP status code, if any.
    pub fn status(&self) -> Option<u16> {
        match self {
            ClientError::RateLimited { .. } => Some(429),
            ClientError::Server { status, .. } | ClientError::Http { status, .. } => Some(*status),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relogin_required_detected_through_other() {
        let e = ClientError::Other(anyhow::anyhow!(crate::AuthError::ReloginRequired(
            "refresh_token expired".into()
        )));
        assert!(e.is_relogin_required());
        assert!(!ClientError::Other(anyhow::anyhow!("some other error")).is_relogin_required());
        assert!(!ClientError::Http { status: 400, body: "bad".into() }.is_relogin_required());
    }
}
