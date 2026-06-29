//! Streaming chat example: prints text deltas as they arrive instead of
//! collecting the full response. First run triggers device-code login.
//!
//! Run:
//!   cargo run --example streaming_chat -- "Ask me something that needs a long answer."

use std::io::Write;
use std::process::ExitCode;

use chatgpt_oauth::{SendOptions, StreamEvent, ensure_logged_in, open_event_stream};
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> ExitCode {
    let prompt = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    if prompt.trim().is_empty() {
        eprintln!("usage: cargo run --example streaming_chat -- <your message>");
        return ExitCode::from(2);
    }

    if let Err(e) = ensure_logged_in().await {
        eprintln!("login failed: {e:#}");
        return ExitCode::FAILURE;
    }

    let opts = SendOptions::default();
    let stream = match open_event_stream(&prompt, &opts).await {
        Ok(s) => s,
        Err(e) => {
            eprintln!("failed to open stream: {e}");
            return ExitCode::FAILURE;
        }
    };
    // Stream is not Unpin (internal async unfold); pin before .next().
    let mut stream = Box::pin(stream);

    let mut printed_any = false;
    while let Some(ev) = stream.next().await {
        let ev = match ev {
            Ok(v) => v,
            Err(e) => {
                eprintln!("\nstream error: {e}");
                return ExitCode::FAILURE;
            }
        };
        match ev {
            StreamEvent::TextDelta(delta) => {
                print!("{delta}");
                let _ = std::io::stdout().flush();
                printed_any = true;
            }
            StreamEvent::Failed(err) => {
                eprintln!("\nresponse failed: {err}");
                return ExitCode::FAILURE;
            }
            StreamEvent::Incomplete(detail) => {
                eprintln!("\nresponse incomplete: {detail}");
                return ExitCode::FAILURE;
            }
            _ => {}
        }
    }

    if printed_any {
        println!();
        ExitCode::SUCCESS
    } else {
        eprintln!("(no text deltas)");
        ExitCode::FAILURE
    }
}
