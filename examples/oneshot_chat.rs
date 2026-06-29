//! One-shot chat example: send a single message and print the response. No
//! history, no REPL. First run triggers device-code login; tokens are cached
//! in ~/.codex/auth.json.
//!
//! Run:
//!   cargo run --example oneshot_chat -- "Introduce yourself in one sentence."
//!
//! Env: CODEX_DEFAULT_MODEL overrides the default model.

use std::process::ExitCode;

use chatgpt_oauth::{SendOptions, ensure_logged_in, send_message};

#[tokio::main]
async fn main() -> ExitCode {
    let prompt = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    if prompt.trim().is_empty() {
        eprintln!("usage: cargo run --example oneshot_chat -- <your message>");
        return ExitCode::from(2);
    }

    if let Err(e) = ensure_logged_in().await {
        eprintln!("login failed: {e:#}");
        return ExitCode::FAILURE;
    }

    // send_message handles creds, 401 refresh, and retries internally.
    let opts = SendOptions::default();
    match send_message(&prompt, &opts).await {
        Ok(response) => {
            let text = response.text();
            if text.is_empty() {
                eprintln!("(no text in response)");
                ExitCode::FAILURE
            } else {
                println!("{text}");
                ExitCode::SUCCESS
            }
        }
        Err(e) => {
            eprintln!("request failed: {e}");
            ExitCode::FAILURE
        }
    }
}
