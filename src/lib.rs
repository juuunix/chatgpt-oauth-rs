//! chatgpt-oauth — pure low-level client for the ChatGPT backend, using a
//! ChatGPT subscription OAuth token (not a paid API key).

// Modules are private; the `pub use` re-exports below are the only public boundary.
mod auth;
mod client;
mod error;
mod event;
mod input;

// Diagnostic raw-response capture; only compiled under feature = "capture".
#[cfg(feature = "capture")]
pub mod capture;

pub use auth::{
    AuthError, CodexCredentials, DeviceCodePrompt, auth_path, device_code_login,
    device_code_login_with, ensure_logged_in, is_access_token_expiring, is_relogin_required,
    load_codex_cli_tokens, resolve_credentials, resolve_credentials_after_401,
    save_codex_cli_tokens_locked, validate_base_url, validate_token_destination,
};
pub use client::{
    Model, RateLimit, RateWindow, SendOptions, Usage, extract_text, fetch_usage, list_models,
    open_stream, open_stream_with_input, send_message, send_with_input,
};
pub use error::ClientError;
// Typed event layer (additive over open_stream).
pub use event::{
    GeneratedImage, Response, StreamEvent, TokenUsage, ToolCall, WebSearch, open_event_stream,
    open_event_stream_with_input,
};
// Typed input builders.
pub use input::{InputItem, Tool};
