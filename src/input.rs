//! Typed builders for request input — `tools` declarations and the `input`
//! array, instead of raw `json!`. Each returns a `Value` for the wire.

use serde_json::{Value, json};

/// Builder for `SendOptions.tools` declarations.
pub struct Tool;

impl Tool {
    /// Custom function tool; `parameters` is a JSON Schema.
    pub fn function(name: impl Into<String>, parameters: Value) -> Value {
        json!({ "type": "function", "name": name.into(), "parameters": parameters })
    }

    pub fn function_described(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
    ) -> Value {
        json!({
            "type": "function",
            "name": name.into(),
            "description": description.into(),
            "parameters": parameters,
        })
    }

    /// Server built-in web_search (accepted by this backend).
    pub fn web_search() -> Value {
        json!({ "type": "web_search" })
    }

    /// Server built-in image_generation (accepted by this backend).
    pub fn image_generation() -> Value {
        json!({ "type": "image_generation" })
    }
}

/// Builder for `input` array items (multiturn / tool result feedback).
pub struct InputItem;

impl InputItem {
    /// User message; content type is `input_text`.
    pub fn user(text: impl Into<String>) -> Value {
        json!({ "role": "user", "content": [{ "type": "input_text", "text": text.into() }] })
    }

    /// Prior assistant turn for history replay. This backend is `store:false`, so
    /// past responses must be fed back in. Note content type is `output_text`
    /// (not `input_text` like user) — the builder fixes this easy-to-miss asymmetry.
    pub fn assistant(text: impl Into<String>) -> Value {
        json!({ "role": "assistant", "content": [{ "type": "output_text", "text": text.into() }] })
    }

    /// Custom tool result feedback; `call_id` matches the model's function_call.
    pub fn function_output(call_id: impl Into<String>, output: impl Into<String>) -> Value {
        json!({ "type": "function_call_output", "call_id": call_id.into(), "output": output.into() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_builders_shape() {
        assert_eq!(Tool::web_search()["type"], "web_search");
        assert_eq!(Tool::image_generation()["type"], "image_generation");
        let f = Tool::function("get_weather", json!({"type":"object"}));
        assert_eq!(f["type"], "function");
        assert_eq!(f["name"], "get_weather");
        let fd = Tool::function_described("x", "desc", json!({}));
        assert_eq!(fd["description"], "desc");
    }

    #[test]
    fn input_item_shape() {
        let u = InputItem::user("hi");
        assert_eq!(u["role"], "user");
        assert_eq!(u["content"][0]["type"], "input_text");
        assert_eq!(u["content"][0]["text"], "hi");
        // assistant uses output_text, unlike user.
        let a = InputItem::assistant("hello");
        assert_eq!(a["role"], "assistant");
        assert_eq!(a["content"][0]["type"], "output_text");
        assert_eq!(a["content"][0]["text"], "hello");
        let o = InputItem::function_output("c1", "{\"temp\":21}");
        assert_eq!(o["type"], "function_call_output");
        assert_eq!(o["call_id"], "c1");
        assert_eq!(o["output"], "{\"temp\":21}");
    }
}
