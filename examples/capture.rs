//! Raw endpoint-response capture tool: dumps each request's response as
//! unparsed JSON, so parser fixes can be based on what the API actually
//! returns. Focuses on surfaces we don't control (server builtin tools,
//! endpoint shapes). Requires the `capture` feature.
//!
//! Scenarios: usage, models, responses [--web|--image] <msg>, builtin-probe,
//! device-code. No args or `all` runs every scenario. Results auto-saved to
//! captures/<date>/<scenario>.json; progress/summary on stderr.
//!
//! Run:
//!   cargo run --example capture --features capture
//!   cargo run --example capture --features capture -- usage
//!   cargo run --example capture --features capture -- responses --image
//!   cargo run --example capture --features capture -- builtin-probe
//!
//! File format: { "meta": {scenario, captured_at_unix}, "capture": <body> }
//! Note: responses/builtin-probe consume tokens; refresh is intentionally
//! excluded since it rotates refresh_token and can break saved auth.

use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use chatgpt_oauth::{SendOptions, capture, device_code_login, load_codex_cli_tokens};
use serde_json::{Value, json};

/// Scenarios run by "all": (name, default prompt).
const ALL_SCENARIOS: &[(&str, &str)] = &[
    ("usage", ""),
    ("models", ""),
    ("responses-text", "Introduce yourself in one sentence."),
    ("responses-web", "Web-search and give me one line of today's top news."),
    ("responses-image", "Draw a small cat icon."),
    ("builtin-probe", ""),
    ("device-code", ""),
];

#[tokio::main]
async fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let scenario = args.first().cloned().unwrap_or_else(|| "all".into());

    let plan: Vec<(String, String)> = match scenario.as_str() {
        "all" => ALL_SCENARIOS.iter().map(|(n, p)| (n.to_string(), p.to_string())).collect(),
        "usage" | "models" | "builtin-probe" | "device-code" => {
            vec![(scenario.clone(), String::new())]
        }
        "responses" => {
            let rest = &args[1..];
            let flag = rest.first().map(|s| s.as_str()).unwrap_or("");
            let (name, skip, default_prompt) = match flag {
                "--web" => ("responses-web", 1, "Web-search and give me one line of today's top news."),
                "--image" => ("responses-image", 1, "Draw a small cat icon."),
                _ => ("responses-text", 0, "Introduce yourself in one sentence."),
            };
            let prompt = rest[skip..].join(" ");
            let prompt = if prompt.trim().is_empty() { default_prompt.to_string() } else { prompt };
            vec![(name.to_string(), prompt)]
        }
        _ => {
            eprintln!(
                "usage: cargo run --example capture --features capture -- [all|usage|responses [--web|--image] [prompt]|builtin-probe|device-code]\n  No args means all (capture everything). Results auto-saved to captures/<date>/<scenario>.json."
            );
            return ExitCode::from(2);
        }
    };

    // Everything except device-code requires login.
    let needs_login = plan.iter().any(|(n, _)| n != "device-code");
    if needs_login && let Err(e) = ensure_login().await {
        return fail(e);
    }

    let dir = format!("captures/{}", capture::today_utc());
    if let Err(e) = std::fs::create_dir_all(&dir) {
        return fail(format!("failed to create {dir}: {e}"));
    }

    let captured_at_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    eprintln!("▶ capture start → {dir}/ ({} scenarios)\n", plan.len());
    let mut ok = 0;
    let mut fail_n = 0;
    for (name, prompt) in &plan {
        let result = run_one(name, prompt).await;
        let (cap_val, status) = match result {
            Ok(v) => (v, "OK".to_string()),
            Err(e) => (json!({ "error": e }), "ERROR".to_string()),
        };
        let doc = json!({
            "meta": { "scenario": name, "captured_at_unix": captured_at_unix },
            "capture": cap_val,
        });
        let path = format!("{dir}/{name}.json");
        match std::fs::write(&path, serde_json::to_string_pretty(&doc).unwrap_or_default()) {
            Ok(()) => {
                let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                eprintln!("  {status:5}  {path}  ({bytes} bytes)");
                if status == "OK" { ok += 1 } else { fail_n += 1 }
            }
            Err(e) => {
                eprintln!("  WRITE-FAIL  {path}: {e}");
                fail_n += 1;
            }
        }
    }
    eprintln!("\ndone: {ok} OK, {fail_n} failed → {dir}/");
    if fail_n > 0 { ExitCode::FAILURE } else { ExitCode::SUCCESS }
}

/// Dispatch a scenario name to its capture call.
async fn run_one(name: &str, prompt: &str) -> Result<Value, String> {
    let to_val = |c: capture::RawCapture| serde_json::to_value(c).unwrap_or(Value::Null);
    match name {
        "usage" => capture::usage_raw().await.map(to_val).map_err(|e| format!("{e:#}")),
        "models" => capture::models_raw().await.map(to_val).map_err(|e| format!("{e:#}")),
        "device-code" => capture::device_usercode_raw().await.map(to_val).map_err(|e| format!("{e:#}")),
        "builtin-probe" => builtin_probe().await,
        "responses-text" | "responses-web" | "responses-image" => {
            let tools = match name {
                "responses-web" => vec![json!({ "type": "web_search" })],
                "responses-image" => vec![json!({ "type": "image_generation" })],
                _ => Vec::new(),
            };
            // SendOptions is #[non_exhaustive]: build via default() + field mutation.
            let mut opts = SendOptions::default();
            opts.tools = tools;
            capture::responses_raw(prompt, &opts)
                .await
                .map(|events| json!({ "prompt": prompt, "event_count": events.len(), "events": events }))
                .map_err(|e| format!("{e:#}"))
        }
        other => Err(format!("unknown scenario: {other}")),
    }
}

/// Probe builtin-tool candidates: send each type and record 200 (accepted)
/// vs 4xx (unsupported) to discover what this backend actually accepts.
async fn builtin_probe() -> Result<Value, String> {
    let candidates = [
        "web_search",
        "image_generation",
        "file_search",
        "code_interpreter",
        "computer_use",
        "local_shell",
    ];
    let mut results = Vec::new();
    for t in candidates {
        let entry = match capture::responses_probe_raw(vec![json!({ "type": t })], "hi").await {
            Ok(rc) => json!({
                "tool": t,
                "status": rc.status,
                "accepted": rc.status == 200,
                "body": rc.body,        // placeholder on 200, error body otherwise
            }),
            Err(e) => json!({ "tool": t, "error": format!("{e:#}") }),
        };
        results.push(entry);
    }
    Ok(json!({ "probed": results }))
}

async fn ensure_login() -> Result<(), String> {
    match load_codex_cli_tokens() {
        Ok(Some(_)) => Ok(()),
        Ok(None) => {
            eprintln!("no saved token — starting device login...");
            device_code_login().await.map(|_| ()).map_err(|e| format!("login failed: {e:#}"))
        }
        Err(e) => Err(format!("failed to read token: {e:#}")),
    }
}

fn fail(msg: String) -> ExitCode {
    eprintln!("{msg}");
    ExitCode::FAILURE
}
