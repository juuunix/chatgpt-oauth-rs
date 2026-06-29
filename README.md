# chatgpt-oauth-rs

A **low-level Rust client** that calls the ChatGPT backend
(`chatgpt.com/backend-api/codex/responses`) directly using a ChatGPT
**subscription OAuth token** (not a paid API key).

It shares the token store (`~/.codex/auth.json`) with the official Codex CLI, so
logging in here also logs you in for the Codex CLI, and vice versa.

> **SDK-style only — no chat session, no history, no REPL.**
> Session and history management is intentionally left to the caller. This crate
> only owns "one request → one response"; for multi-turn you build the `input`
> array yourself and resend it whole on every call (see below).

## Requirements

- **Rust 1.85+** (edition 2024)
- A tokio runtime (every call is async)

## Install

Not published to crates.io yet (0.1.0). Use a git dependency:

```toml
[dependencies]
chatgpt-oauth-rs = { git = "https://github.com/juuunix/chatgpt-oauth-rs" }  # imported as `chatgpt_oauth`
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
anyhow = "1"           # for the Result<()> + ? error propagation in the examples
futures-util = "0.3"   # for the streaming API (.next())
serde_json = "1"       # for building input arrays / tool specs directly
```

## Quickstart (one-shot)

```rust
use chatgpt_oauth::{SendOptions, ensure_logged_in, send_message};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Reuse a saved token, or run device-code login (URL/code printed to stdout).
    ensure_logged_in().await?;

    let opts = SendOptions::new("gpt-5.3-codex");
    let resp = send_message("Introduce yourself in one sentence.", &opts).await?;
    println!("{}", resp.text());
    Ok(())
}
```

`SendOptions` is `#[non_exhaustive]`, so build it via `new(model)`/`default()` +
field mutation instead of a struct literal:

```rust
let mut opts = SendOptions::new("gpt-5.3-codex");
opts.reasoning_effort = Some("high".into());     // "low" | "medium" | "high"
opts.instructions = "You are a concise assistant.".into();
```

## Streaming

Receive token deltas as they arrive instead of buffering the whole response.
Match on `StreamEvent` instead of memorizing magic strings:

```rust
use chatgpt_oauth::{SendOptions, StreamEvent, ensure_logged_in, open_event_stream};
use futures_util::StreamExt;

ensure_logged_in().await?;
let opts = SendOptions::new("gpt-5.3-codex");
let mut stream = Box::pin(open_event_stream("Ask me something long.", &opts).await?);

while let Some(ev) = stream.next().await {
    match ev? {
        StreamEvent::TextDelta(d)   => print!("{d}"),
        StreamEvent::Failed(e)      => eprintln!("failed: {e}"),
        StreamEvent::Incomplete(d)  => eprintln!("incomplete: {d}"),
        _ => {}   // created/in_progress/tool-progress/unknown new events flow through as Other
    }
}
```

## Multi-turn — the caller owns the history

This backend runs with `store:false`, so **the server keeps no memory of the
conversation.** To preserve context you stack prior turns in the `input` array
and **resend the whole thing on every call.** Build it with `InputItem` instead
of raw JSON:

```rust
use chatgpt_oauth::{InputItem, SendOptions, ensure_logged_in, send_with_input};
use serde_json::Value;

ensure_logged_in().await?;
let opts = SendOptions::new("gpt-5.3-codex");

let mut history: Vec<Value> = vec![InputItem::user("My name is Jun.")];
let r1 = send_with_input(Value::Array(history.clone()), &opts).await?;
println!("{}", r1.text());

history.push(InputItem::assistant(r1.text()));      // feed the model's reply back in
history.push(InputItem::user("What did I say my name was?"));
let r2 = send_with_input(Value::Array(history.clone()), &opts).await?;
println!("{}", r2.text());   // → "Jun"
```

> `InputItem::user` uses `input_text`, `InputItem::assistant` uses `output_text`
> (an API asymmetry). The builders pin this distinction down for you.

> ⚠️ **This example keeps a text-only history.** Feeding back `assistant(r1.text())`
> preserves only the model's *text* answer; non-text `output[]` items such as
> reasoning, tool calls, and builtin-tool output are dropped. That is fine for
> plain text chat, but multi-turn that uses reasoning effort or tools must preserve
> and feed back the prior response's `output[]` items (e.g. `tc.to_input_item()`,
> `reasoning.encrypted_content`) to keep context and reasoning continuity.

## Custom tools (function calling)

Declare (`Tool`) → model emits a `ToolCall` → you run it → feed the result back:

```rust
use chatgpt_oauth::{InputItem, SendOptions, Tool, ToolCall};
use serde_json::json;

let mut opts = SendOptions::new("gpt-5.3-codex");
opts.tools = vec![Tool::function_described(
    "get_weather", "Current weather for a city",
    json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}),
)];

// ... when you get a ToolCall from the stream/response:
// input.push(tc.to_input_item());                              // echo the function_call
// input.push(InputItem::function_output(&tc.call_id, result)); // the execution result
```

Server builtin tools are supported too: `Tool::web_search()`,
`Tool::image_generation()`. Completed results arrive as
`StreamEvent::WebSearchCall` / `ImageGenerated` (or `Response::web_searches()` /
`images()`). Decoding image base64 is the caller's job (low-level client).

Full working example: [`examples/agent_loop.rs`](examples/agent_loop.rs).

## Error handling

Every call returns a typed `ClientError` so you can branch on it programmatically:

```rust
match send_message(msg, &opts).await {
    Ok(r) => println!("{}", r.text()),
    // token expired and auto-refresh also failed → re-login and retry
    Err(e) if e.is_relogin_required() => { chatgpt_oauth::device_code_login().await?; }
    Err(e) => eprintln!("failed (status={:?}): {e}", e.status()),
}
```

- Provides `is_retryable()` / `is_retryable_non_idempotent()` / `retry_after()` / `status()`.
- **Retry policy**: GETs (`list_models`/`fetch_usage`) retry broadly. The POST to
  `/responses` only retries errors where the server provably never received the
  request (connect failure / 429), to **avoid double tool runs and double billing.**
- The library already does backoff retries within `opts.max_retries` (default 2).

## Auth / environment variables

Tokens are written atomically to `~/.codex/auth.json` with `0600` permissions. To
intercept the prompt output (GUI/TUI/bot), use `device_code_login_with` — if the
callback returns `Err` (e.g. delivering the code failed), polling never starts and
the error is surfaced immediately:

```rust
device_code_login_with(|p| {
    // show p.user_code / p.verification_url in your own UI / channel
    my_ui.show_login_code(p.user_code)?;   // on failure, skip polling and error out
    Ok(())
}).await?;
```

| Variable | Purpose |
|---|---|
| `CODEX_HOME` | Override the auth.json location (**absolute path** only) |
| `CODEX_DEFAULT_MODEL` | Default model for `SendOptions::default()` (falls back to `gpt-5.3-codex`) |
| `CODEX_BASE_URL` | Override the API base URL (trusted-host validation still applies) |
| `CODEX_ALLOW_INSECURE_BASE_URL` | Bypass host/https validation (self-hosting/testing). Even so, **disk-loaded tokens** are sent only to trusted hosts/loopback over https |

**Security**: `base_url` is validated with a URL parser that allows only
`chatgpt.com` / `openai.com` and blocks evasion tricks like
`https://evil.com#@chatgpt.com`. Tokens are shown as `[redacted]` in `Debug` output.

## Examples

```bash
cargo run --example oneshot_chat   -- "Introduce yourself in one sentence."
cargo run --example streaming_chat -- "Ask me something that needs a long answer."
cargo run --example agent_loop     -- "What's the weather in Seoul?"
```

## Diagnostic capture (optional)

With the `capture` feature, you can dump each endpoint's **raw, pre-parse response**
to inspect backend changes by eye (off by default, no effect on the public API):

```bash
cargo run --example capture --features capture
```

## License

(TBD)
