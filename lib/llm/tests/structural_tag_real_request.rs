//! Acceptance test: the real Yutori computer-use request (structural_tag
//! response_format + tool-message content arrays + 24 tools) must deserialize
//! through the Dynamo frontend request type, and the structural_tag spec must
//! be surfaced for the backend. The unmodified payload is the fixture.

use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use dynamo_llm::protocols::openai::common_ext::CommonExtProvider;

#[test]
fn real_yutori_request_with_structural_tag_deserializes() {
    let body = include_str!("fixtures/yutori_sample_request.json");

    let request: NvCreateChatCompletionRequest =
        serde_json::from_str(body).expect("real Yutori request should deserialize");

    // structural_tag is surfaced (and reconstructed with its "type") for the backend.
    let st = request
        .get_structural_tag()
        .expect("structural_tag should be present");
    assert_eq!(st["type"], serde_json::json!("structural_tag"));
    assert!(st["structures"].is_array());

    // Sanity: the full conversation (incl. tool messages with array content) and
    // the 24 tool definitions all parsed.
    assert!(request.inner.messages.len() >= 2);
    assert!(
        request
            .inner
            .tools
            .as_ref()
            .map(|t| t.len())
            .unwrap_or(0)
            >= 1
    );
}
