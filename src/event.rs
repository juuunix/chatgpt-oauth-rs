//! Thin typed layer over `/responses` — streaming `StreamEvent` + single `Response`.
//!
//! Design: don't mirror the full schema. Event kinds are an open set, so only
//! high-value ones (text delta, custom tool calls) are typed; everything else is
//! `Value` under `Other` for forward-compat (new OpenAI events don't break us).
//! Raw `open_stream` remains available; this layer is additive.

use futures_util::{Stream, StreamExt};
use serde_json::{Value, json};

use crate::client::{SendOptions, extract_text, open_stream, open_stream_with_input};
use crate::error::ClientError;

/// A custom function tool call from the model (we execute and feed back).
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub call_id: String,
    pub name: String,
    /// Raw JSON string; parse via `arguments_json()`.
    pub arguments: String,
}

impl ToolCall {
    /// `ToolCall` if item is a `function_call`, else None.
    pub fn from_item(item: &Value) -> Option<ToolCall> {
        if item.get("type").and_then(|t| t.as_str()) != Some("function_call") {
            return None;
        }
        let call_id = str_of(item, "call_id");
        let name = str_of(item, "name");
        // Malformed without call_id/name → None, to avoid feeding back an unmatchable result.
        if call_id.is_empty() || name.is_empty() {
            return None;
        }
        Some(ToolCall { call_id, name, arguments: str_of(item, "arguments") })
    }

    pub fn arguments_json(&self) -> Option<Value> {
        serde_json::from_str(&self.arguments).ok()
    }

    /// function_call echo item for the next turn's input (store:false replay).
    pub fn to_input_item(&self) -> Value {
        json!({
            "type": "function_call",
            "name": self.name,
            "call_id": self.call_id,
            "arguments": self.arguments,
        })
    }
}

/// Completed server built-in web_search. Only the query reaches us; the answer
/// the model builds from results arrives as TextDelta.
#[derive(Debug, Clone)]
pub struct WebSearch {
    pub id: String,
    pub status: String,
    pub query: Option<String>,
    pub queries: Vec<String>,
}

/// Completed server built-in image_generation.
#[derive(Debug, Clone)]
pub struct GeneratedImage {
    pub id: String,
    pub status: String,
    /// Raw base64 (usually PNG); consumer decodes.
    pub result_b64: Option<String>,
    pub revised_prompt: Option<String>,
    pub size: Option<String>,
    pub quality: Option<String>,
    pub output_format: Option<String>,
}

impl GeneratedImage {
    /// `GeneratedImage` if item is an `image_generation_call`, else None.
    pub fn from_item(item: &Value) -> Option<GeneratedImage> {
        if item.get("type").and_then(|t| t.as_str()) != Some("image_generation_call") {
            return None;
        }
        Some(GeneratedImage {
            id: str_of(item, "id"),
            status: str_of(item, "status"),
            result_b64: item.get("result").and_then(|v| v.as_str()).map(String::from),
            revised_prompt: item.get("revised_prompt").and_then(|v| v.as_str()).map(String::from),
            size: item.get("size").and_then(|v| v.as_str()).map(String::from),
            quality: item.get("quality").and_then(|v| v.as_str()).map(String::from),
            output_format: item.get("output_format").and_then(|v| v.as_str()).map(String::from),
        })
    }

}

impl WebSearch {
    /// `WebSearch` if item is a `web_search_call`, else None.
    pub fn from_item(item: &Value) -> Option<WebSearch> {
        if item.get("type").and_then(|t| t.as_str()) != Some("web_search_call") {
            return None;
        }
        let action = item.get("action");
        Some(WebSearch {
            id: str_of(item, "id"),
            status: str_of(item, "status"),
            query: action.and_then(|a| a.get("query")).and_then(|v| v.as_str()).map(String::from),
            queries: action
                .and_then(|a| a.get("queries"))
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|q| q.as_str().map(String::from)).collect())
                .unwrap_or_default(),
        })
    }
}

fn str_of(v: &Value, key: &str) -> String {
    v.get(key).and_then(|x| x.as_str()).unwrap_or("").to_string()
}

/// Typed stream event. Only high-value kinds are typed; the rest fall to
/// `Other { kind, raw }` for forward-compat.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// `response.output_text.delta`.
    TextDelta(String),
    /// Completed custom function tool call (`output_item.done`, function_call).
    ToolCall(ToolCall),
    WebSearchCall(WebSearch),
    ImageGenerated(GeneratedImage),
    /// `response.completed` — final `response` object.
    Completed(Value),
    /// `response.failed` — terminal failure, error payload.
    Failed(Value),
    /// `response.incomplete` — terminal incomplete, incomplete_details payload.
    Incomplete(Value),
    /// Everything else (lifecycle, built-in tool progress, unknown new events).
    Other { kind: String, raw: Value },
}

impl StreamEvent {
    pub fn from_event(ev: &Value) -> StreamEvent {
        let kind = ev.get("type").and_then(|t| t.as_str()).unwrap_or("");
        match kind {
            "response.output_text.delta" => {
                StreamEvent::TextDelta(
                    ev.get("delta").and_then(|d| d.as_str()).unwrap_or("").to_string(),
                )
            }
            // Even on completed, a terminal inner status maps to Failed/Incomplete
            // so failure isn't mistaken for success.
            "response.completed" => {
                let resp = ev.get("response").cloned().unwrap_or(Value::Null);
                match resp.get("status").and_then(|s| s.as_str()).unwrap_or("") {
                    "failed" | "cancelled" | "expired" => {
                        StreamEvent::Failed(resp.get("error").cloned().unwrap_or(Value::Null))
                    }
                    "incomplete" => StreamEvent::Incomplete(
                        resp.get("incomplete_details").cloned().unwrap_or(Value::Null),
                    ),
                    _ => StreamEvent::Completed(resp),
                }
            }
            "response.failed" => StreamEvent::Failed(
                ev.get("response")
                    .and_then(|r| r.get("error"))
                    .or_else(|| ev.get("error"))
                    .cloned()
                    .unwrap_or(Value::Null),
            ),
            "response.incomplete" => StreamEvent::Incomplete(
                ev.get("response")
                    .and_then(|r| r.get("incomplete_details"))
                    .or_else(|| ev.get("incomplete_details"))
                    .cloned()
                    .unwrap_or(Value::Null),
            ),
            // Branch on item.type via the shared from_item helpers; rest → Other.
            "response.output_item.done" => {
                let item = ev.get("item").cloned().unwrap_or(Value::Null);
                if let Some(tc) = ToolCall::from_item(&item) {
                    StreamEvent::ToolCall(tc)
                } else if let Some(ws) = WebSearch::from_item(&item) {
                    StreamEvent::WebSearchCall(ws)
                } else if let Some(img) = GeneratedImage::from_item(&item) {
                    StreamEvent::ImageGenerated(img)
                } else {
                    StreamEvent::Other { kind: kind.to_string(), raw: ev.clone() }
                }
            }
            other => StreamEvent::Other { kind: other.to_string(), raw: ev.clone() },
        }
    }
}

/// Typed version of `open_stream`, yielding `StreamEvent`.
pub async fn open_event_stream(
    user_message: &str,
    opts: &SendOptions,
) -> Result<impl Stream<Item = Result<StreamEvent, ClientError>>, ClientError> {
    Ok(open_stream(user_message, opts)
        .await?
        .map(|r| r.map(|v| StreamEvent::from_event(&v))))
}

/// Typed version of `open_stream_with_input` (multiturn / tool result feedback).
pub async fn open_event_stream_with_input(
    input: Value,
    opts: &SendOptions,
) -> Result<impl Stream<Item = Result<StreamEvent, ClientError>>, ClientError> {
    Ok(open_stream_with_input(input, opts)
        .await?
        .map(|r| r.map(|v| StreamEvent::from_event(&v))))
}

/// Per-request token usage (`response.completed.usage`); distinct from quota (`fetch_usage`).
#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    pub input: u64,
    pub output: u64,
    pub total: u64,
    pub cached: u64,
    pub reasoning: u64,
}

impl TokenUsage {
    pub fn from_response(response: &Value) -> Option<TokenUsage> {
        let u = response.get("usage")?;
        let n = |path: &[&str]| -> u64 {
            let mut cur = u;
            for k in path {
                match cur.get(k) {
                    Some(v) => cur = v,
                    None => return 0,
                }
            }
            cur.as_u64().unwrap_or(0)
        };
        Some(TokenUsage {
            input: n(&["input_tokens"]),
            output: n(&["output_tokens"]),
            total: n(&["total_tokens"]),
            cached: n(&["input_tokens_details", "cached_tokens"]),
            reasoning: n(&["output_tokens_details", "reasoning_tokens"]),
        })
    }
}

/// Typed response for a single `send_message`; mirror of streaming StreamEvent.
#[derive(Debug, Clone)]
pub struct Response {
    raw: Value,
}

impl Response {
    pub(crate) fn new(raw: Value) -> Response {
        Response { raw }
    }

    pub fn text(&self) -> String {
        extract_text(&self.raw)
    }

    fn output_items(&self) -> &[Value] {
        self.raw.get("output").and_then(|o| o.as_array()).map(|v| v.as_slice()).unwrap_or(&[])
    }

    pub fn tool_calls(&self) -> Vec<ToolCall> {
        self.output_items().iter().filter_map(ToolCall::from_item).collect()
    }

    pub fn web_searches(&self) -> Vec<WebSearch> {
        self.output_items().iter().filter_map(WebSearch::from_item).collect()
    }

    pub fn images(&self) -> Vec<GeneratedImage> {
        self.output_items().iter().filter_map(GeneratedImage::from_item).collect()
    }

    pub fn usage(&self) -> Option<TokenUsage> {
        TokenUsage::from_response(&self.raw)
    }

    /// Raw `response` Value (escape for fields not in the typed layer).
    pub fn raw(&self) -> &Value {
        &self.raw
    }

    pub fn into_raw(self) -> Value {
        self.raw
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn classifies_text_delta() {
        let ev = json!({"type":"response.output_text.delta","delta":"hi"});
        match StreamEvent::from_event(&ev) {
            StreamEvent::TextDelta(t) => assert_eq!(t, "hi"),
            other => panic!("expected TextDelta, got {other:?}"),
        }
    }

    #[test]
    fn classifies_function_call_tool_call() {
        let ev = json!({
            "type":"response.output_item.done",
            "item":{"type":"function_call","name":"get_weather",
                    "call_id":"call_123","arguments":"{\"city\":\"Seoul\"}","status":"completed"}
        });
        match StreamEvent::from_event(&ev) {
            StreamEvent::ToolCall(tc) => {
                assert_eq!(tc.name, "get_weather");
                assert_eq!(tc.call_id, "call_123");
                assert_eq!(tc.arguments_json().unwrap()["city"], "Seoul");
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn completed_with_failed_status_maps_to_failed() {
        let ev = json!({"type":"response.completed","response":{"status":"failed","error":{"message":"boom"}}});
        assert!(matches!(StreamEvent::from_event(&ev), StreamEvent::Failed(_)));
        let inc = json!({"type":"response.completed","response":{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}});
        assert!(matches!(StreamEvent::from_event(&inc), StreamEvent::Incomplete(_)));
    }

    #[test]
    fn malformed_function_call_is_not_toolcall() {
        let ev = json!({"type":"response.output_item.done","item":{"type":"function_call","arguments":"{}"}});
        assert!(matches!(StreamEvent::from_event(&ev), StreamEvent::Other { .. }));
        assert!(ToolCall::from_item(&json!({"type":"function_call","name":"x"})).is_none());
    }

    #[test]
    fn message_item_done_is_other_not_toolcall() {
        let ev = json!({"type":"response.output_item.done","item":{"type":"message","content":[]}});
        assert!(matches!(StreamEvent::from_event(&ev), StreamEvent::Other { .. }));
    }

    #[test]
    fn classifies_web_search_call() {
        let ev = json!({"type":"response.output_item.done","item":{
            "type":"web_search_call","id":"ws_1","status":"completed",
            "action":{"type":"search","query":"latest news","queries":["a","b"]}}});
        match StreamEvent::from_event(&ev) {
            StreamEvent::WebSearchCall(w) => {
                assert_eq!(w.id, "ws_1");
                assert_eq!(w.query.as_deref(), Some("latest news"));
                assert_eq!(w.queries, vec!["a".to_string(), "b".to_string()]);
            }
            other => panic!("expected WebSearchCall, got {other:?}"),
        }
    }

    #[test]
    fn classifies_image_generated() {
        let ev = json!({"type":"response.output_item.done","item":{
            "type":"image_generation_call","id":"ig_1","status":"completed",
            "result":"aGk=","output_format":"png","size":"1254x1254","quality":"low",
            "revised_prompt":"a cat icon"}});
        match StreamEvent::from_event(&ev) {
            StreamEvent::ImageGenerated(img) => {
                assert_eq!(img.id, "ig_1");
                assert_eq!(img.output_format.as_deref(), Some("png"));
                assert_eq!(img.revised_prompt.as_deref(), Some("a cat icon"));
                assert_eq!(img.result_b64.as_deref(), Some("aGk="));
            }
            other => panic!("expected ImageGenerated, got {other:?}"),
        }
    }

    #[test]
    fn classifies_completed_and_terminals() {
        let c = json!({"type":"response.completed","response":{"status":"completed","usage":{"total_tokens":33}}});
        match StreamEvent::from_event(&c) {
            StreamEvent::Completed(r) => {
                assert_eq!(r["usage"]["total_tokens"], 33);
                assert_eq!(TokenUsage::from_response(&r).unwrap().total, 33);
            }
            other => panic!("expected Completed, got {other:?}"),
        }
        let f = json!({"type":"response.failed","response":{"error":{"message":"boom"}}});
        assert!(matches!(StreamEvent::from_event(&f), StreamEvent::Failed(_)));
        let i = json!({"type":"response.incomplete","response":{"incomplete_details":{"reason":"max_output_tokens"}}});
        assert!(matches!(StreamEvent::from_event(&i), StreamEvent::Incomplete(_)));
    }

    #[test]
    fn response_typed_accessors() {
        let resp = Response::new(json!({
            "status":"completed",
            "output":[
                {"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]},
                {"type":"function_call","name":"get_weather","call_id":"c1","arguments":"{\"city\":\"Seoul\"}"}
            ],
            "usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15,
                     "input_tokens_details":{"cached_tokens":2},
                     "output_tokens_details":{"reasoning_tokens":3}}
        }));
        assert_eq!(resp.text(), "hi");
        let calls = resp.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].call_id, "c1");
        let u = resp.usage().unwrap();
        assert_eq!((u.input, u.output, u.total, u.cached, u.reasoning), (10, 5, 15, 2, 3));
        let echo = calls[0].to_input_item();
        assert_eq!(echo["type"], "function_call");
        assert_eq!(echo["call_id"], "c1");
    }

    #[test]
    fn unknown_and_builtin_events_are_other_forward_compat() {
        for kind in [
            "response.web_search_call.searching",
            "response.image_generation_call.partial_image",
            "response.function_call_arguments.delta",
            "response.some_future_event_42",
        ] {
            let ev = json!({"type": kind, "x": 1});
            match StreamEvent::from_event(&ev) {
                StreamEvent::Other { kind: k, raw } => {
                    assert_eq!(k, kind);
                    assert_eq!(raw["x"], 1);
                }
                other => panic!("expected Other for {kind}, got {other:?}"),
            }
        }
    }
}
