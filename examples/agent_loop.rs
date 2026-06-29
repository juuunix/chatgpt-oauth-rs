//! Custom-tool agent loop using only typed inputs/outputs (no raw JSON).
//! Declare tool, model emits ToolCall, we run it and feed the result back,
//! model produces the final text answer.
//!
//! Run:
//!   cargo run --example agent_loop -- "What's the weather in Seoul?"

use std::io::Write;
use std::process::ExitCode;

use chatgpt_oauth::{
    InputItem, SendOptions, StreamEvent, Tool, ToolCall, ensure_logged_in,
    open_event_stream_with_input,
};
use futures_util::StreamExt;
use serde_json::{Value, json};

#[tokio::main]
async fn main() -> ExitCode {
    let prompt = {
        let p = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
        if p.trim().is_empty() { "What's the weather in Seoul?".to_string() } else { p }
    };

    if let Err(e) = ensure_logged_in().await {
        eprintln!("login failed: {e:#}");
        return ExitCode::FAILURE;
    }

    // SendOptions is #[non_exhaustive]: build via default() + field mutation.
    let mut opts = SendOptions::default();
    opts.instructions = "You are a weather assistant. When asked about weather, you MUST call the get_weather tool.".into();
    opts.tools = vec![Tool::function_described(
        "get_weather",
        "Returns the current weather for the given city.",
        json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}),
    )];
    opts.tool_choice = Some(json!("auto"));

    let mut input: Vec<Value> = vec![InputItem::user(&prompt)];

    for turn in 1..=5 {
        let mut stream =
            match open_event_stream_with_input(Value::Array(input.clone()), &opts).await {
                Ok(s) => Box::pin(s),
                Err(e) => {
                    eprintln!("failed to open stream: {e}");
                    return ExitCode::FAILURE;
                }
            };

        let mut text = String::new();
        let mut calls: Vec<ToolCall> = Vec::new();
        while let Some(ev) = stream.next().await {
            match ev {
                Ok(StreamEvent::TextDelta(d)) => {
                    print!("{d}");
                    let _ = std::io::stdout().flush();
                    text.push_str(&d);
                }
                Ok(StreamEvent::ToolCall(tc)) => calls.push(tc),
                Ok(StreamEvent::Failed(e)) => {
                    eprintln!("\nresponse failed: {e}");
                    return ExitCode::FAILURE;
                }
                Ok(_) => {}
                Err(e) => {
                    eprintln!("\nstream error: {e}");
                    return ExitCode::FAILURE;
                }
            }
        }

        // No tool calls means this was the final answer.
        if calls.is_empty() {
            println!();
            return ExitCode::SUCCESS;
        }

        // Run each tool and feed back the echoed call plus its output.
        for tc in &calls {
            let args = tc.arguments_json().unwrap_or(Value::Null);
            let result = run_tool(&tc.name, &args);
            eprintln!("[turn {turn}] {}({}) -> {result}", tc.name, tc.arguments);
            input.push(tc.to_input_item());
            input.push(InputItem::function_output(&tc.call_id, result));
        }
    }

    eprintln!("exceeded max turns");
    ExitCode::FAILURE
}

/// Tool execution stub.
fn run_tool(name: &str, args: &Value) -> String {
    match name {
        "get_weather" => {
            let city = args.get("city").and_then(|c| c.as_str()).unwrap_or("unknown");
            json!({ "city": city, "temp_c": 21, "condition": "clear" }).to_string()
        }
        _ => json!({ "error": "unknown tool" }).to_string(),
    }
}
