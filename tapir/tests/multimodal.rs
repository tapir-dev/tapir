// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! End-to-end tests for the multimodal turn-entry seam: the mixed text+image
//! entry points [`Agent::prompt_with`] / [`Agent::converse_with`] and
//! [`RunHandle::steer_input`]. They prove an image content part rides from the
//! entry point into the provider request, that a steered image lands on the next
//! turn, that an image bound for a non-multimodal model is rejected outright, and
//! that the existing text-only path is unchanged. The wire-level check drives a
//! real `AnthropicProvider` over a recording transport and asserts the recorded
//! HTTP request carries the base64 image block.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use tapir::message::UserInput;
use tapir::{Agent, AgentEvent};
use tapir_provider::{
    AssistantMessage, CompletionOptions, ContentPart, Context,
    Error as ProviderError, FinishReason, ImageSource, MediaType, Message,
    Provider, StreamAccumulator, StreamEvent, StreamEvents, Usage,
};
use tokio::sync::broadcast;

/// A `Provider` that replays one scripted reply per completion and records the
/// content parts of every user turn it saw, per completion, so a test can prove
/// which parts (text and image) reached the provider request and in what order.
/// The last script repeats once the cursor runs past the list, so a parked run
/// that wakes and re-prompts keeps getting an answer.
struct RecordingProvider {
    scripts: Vec<Vec<StreamEvent>>,
    cursor: AtomicUsize,
    user_turns: Arc<Mutex<Vec<Vec<Vec<ContentPart>>>>>,
}

impl RecordingProvider {
    fn new(scripts: Vec<Vec<StreamEvent>>) -> Self {
        Self {
            scripts,
            cursor: AtomicUsize::new(0),
            user_turns: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// A handle onto the per-completion log of user-turn content parts.
    fn user_turns(&self) -> Arc<Mutex<Vec<Vec<Vec<ContentPart>>>>> {
        self.user_turns.clone()
    }

    fn record(&self, ctx: &Context) -> Vec<StreamEvent> {
        let turns: Vec<Vec<ContentPart>> = ctx
            .messages
            .iter()
            .filter_map(|m| match m {
                Message::User { content } => Some(content.clone()),
                _ => None,
            })
            .collect();
        self.user_turns.lock().unwrap().push(turns);
        let i = self.cursor.fetch_add(1, Ordering::SeqCst);
        self.scripts[i.min(self.scripts.len() - 1)].clone()
    }
}

#[async_trait]
impl Provider for RecordingProvider {
    async fn complete(
        &self,
        ctx: &Context,
        _opts: &CompletionOptions,
    ) -> Result<AssistantMessage, ProviderError> {
        Ok(StreamAccumulator::fold(&self.record(ctx)))
    }

    async fn complete_stream(
        &self,
        ctx: &Context,
        _opts: &CompletionOptions,
    ) -> Result<StreamEvents, ProviderError> {
        let events = self.record(ctx);
        Ok(Box::pin(futures_util::stream::iter(
            events.into_iter().map(Ok::<StreamEvent, ProviderError>),
        )))
    }
}

/// A scripted reply that streams `text` and stops (no tool calls).
fn text_script(text: &str) -> Vec<StreamEvent> {
    vec![
        StreamEvent::MessageStart,
        StreamEvent::TextStart { index: 0 },
        StreamEvent::TextDelta {
            index: 0,
            text: text.to_string(),
        },
        StreamEvent::TextEnd { index: 0 },
        StreamEvent::Done {
            finish_reason: FinishReason::Stop,
            usage: Usage::default(),
        },
    ]
}

/// A one-pixel-ish PNG stand-in; only the bytes matter, not that they decode.
const IMAGE_BYTES: &[u8] = b"hi";

/// An image part carrying [`IMAGE_BYTES`] as raw PNG bytes.
fn png_image() -> ImageSource {
    ImageSource::bytes(MediaType::Png, IMAGE_BYTES.to_vec())
}

/// Whether a user turn's parts carry the [`png_image`] part.
fn carries_png(parts: &[ContentPart]) -> bool {
    parts.iter().any(|p| {
        matches!(
            p,
            ContentPart::Image(ImageSource::Bytes { media_type, data })
                if *media_type == MediaType::Png && data == IMAGE_BYTES
        )
    })
}

/// Await the next [`AgentEvent::Idle`], bounded by a timeout so a run that fails
/// to park does not hang the test.
async fn wait_for_idle(events: &mut broadcast::Receiver<AgentEvent>) {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match events.recv().await {
                Ok(AgentEvent::Idle { .. }) => return,
                Ok(AgentEvent::AgentEnd { .. }) => {
                    panic!("run ended before it parked in Idle")
                }
                Ok(_) => continue,
                Err(err) => panic!("event stream closed before Idle: {err}"),
            }
        }
    })
    .await
    .expect("a run must park in Idle within the timeout");
}

#[tokio::test]
async fn prompt_with_carries_an_image_into_the_provider_request() {
    let provider = RecordingProvider::new(vec![text_script("a cat")]);
    let user_turns = provider.user_turns();
    let agent = Agent::builder().provider(provider).build().expect("build");

    let run = agent
        .prompt_with(UserInput::text("what is this?").with_image(png_image()))
        .expect("a hand-supplied provider never rejects an image");
    run.await.expect("run");

    let seen = user_turns.lock().unwrap().clone();
    assert_eq!(seen.len(), 1, "the run ran one turn");
    assert_eq!(
        seen[0][0],
        vec![
            ContentPart::text("what is this?"),
            ContentPart::image(png_image())
        ],
        "the user turn must carry the text then the image, in order"
    );
    assert!(
        carries_png(&seen[0][0]),
        "the image part reached the request"
    );
}

#[tokio::test]
async fn prompt_with_text_only_carries_a_lone_text_part() {
    let provider = RecordingProvider::new(vec![text_script("ok")]);
    let user_turns = provider.user_turns();
    let agent = Agent::builder().provider(provider).build().expect("build");

    // The string path through the new entry point is a single text part, exactly
    // as `prompt` produces — the text-only path is unchanged.
    let run = agent.prompt_with("just text").expect("text never rejects");
    run.await.expect("run");

    let seen = user_turns.lock().unwrap().clone();
    assert_eq!(seen[0][0], vec![ContentPart::text("just text")]);
    assert!(!carries_png(&seen[0][0]));
}

#[tokio::test]
async fn steer_input_appends_an_image_turn_picked_up_next_turn() {
    // Turn 0 answers the opening prompt (a tool-free reply, so the run parks);
    // the steered image lands, and turn 1 sees it.
    let provider = RecordingProvider::new(vec![
        text_script("hi"),
        text_script("i see it"),
    ]);
    let user_turns = provider.user_turns();
    let agent = Agent::builder().provider(provider).build().expect("build");

    let mut events = agent.subscribe();
    let run = agent.converse("start");
    let handle = run.handle();
    let task = tokio::spawn(async move { run.await });

    // Park after the first tool-free reply, then steer an image in.
    wait_for_idle(&mut events).await;
    handle
        .steer_input(UserInput::text("here").with_image(png_image()))
        .expect("a hand-supplied provider never rejects an image steer");
    // Park again after the re-prompted turn that saw the steered image.
    wait_for_idle(&mut events).await;
    handle.finish();

    let reply = tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .expect("run must complete")
        .expect("run task")
        .expect("run");
    assert_eq!(reply.text_content(), "i see it");

    let seen = user_turns.lock().unwrap().clone();
    assert_eq!(seen.len(), 2, "the run ran two turns");
    // Turn 1's context is the opening prompt plus the steered image turn.
    let turn_one = &seen[1];
    assert_eq!(turn_one[0], vec![ContentPart::text("start")]);
    assert_eq!(
        turn_one[1],
        vec![ContentPart::text("here"), ContentPart::image(png_image())],
        "the steered image turn must be appended and picked up next turn"
    );
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn an_image_to_a_non_multimodal_model_is_rejected() {
    use tapir::Error;

    // SAFETY: nextest runs each test in its own process, so this mutation is not
    // observed by any concurrent test.
    unsafe {
        std::env::set_var("OPENAI_API_KEY", "sk-test-offline");
    }
    // `text-embedding-3-small` is the baseline's only text-only model: its input
    // modalities exclude images, so the seam must reject one outright.
    let agent = Agent::builder()
        .model("text-embedding-3-small")
        .build()
        .expect("a known id with an env credential resolves offline");

    let rejected = agent
        .converse_with(UserInput::image(png_image()))
        .err()
        .expect("an image to a non-multimodal model must be rejected");
    assert!(
        matches!(rejected, Error::ImageUnsupported),
        "expected Error::ImageUnsupported, got {rejected:?}"
    );

    // Text alone is fine on the same model — only the image is refused. Abort the
    // spawned run at once so no network call is attempted.
    let run = agent
        .converse_with("plain text is fine")
        .expect("a text-only turn is never rejected");
    run.handle().abort();
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn a_multimodal_model_admits_the_image() {
    // SAFETY: nextest runs each test in its own process, so this mutation is not
    // observed by any concurrent test.
    unsafe {
        std::env::set_var("OPENAI_API_KEY", "sk-test-offline");
    }
    // `gpt-4o` accepts image input, so the seam admits the turn. Abort the spawned
    // run at once, before its turn loop reaches the provider, so no network call
    // is made.
    let agent = Agent::builder().model("gpt-4o").build().expect("build");
    let run = agent
        .converse_with(UserInput::image(png_image()))
        .expect("a multimodal model must admit the image");
    run.handle().abort();
}

/// A full Anthropic message stream: a text block, then usage and stop — enough
/// for the run to settle after one turn.
#[cfg(feature = "anthropic")]
const SAMPLE_STREAM: &str = concat!(
    "event: message_start\n",
    "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"usage\":{\"input_tokens\":7,\"output_tokens\":0}}}\n\n",
    "event: content_block_start\n",
    "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
    "event: content_block_delta\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n",
    "event: content_block_stop\n",
    "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
    "event: message_delta\n",
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n",
    "event: message_stop\n",
    "data: {\"type\":\"message_stop\"}\n\n",
);

/// The record/replay wire check: drive a real `AnthropicProvider` over a
/// recording mock transport through the multimodal entry point, then assert the
/// recorded HTTP request body carries the base64 image block. `IMAGE_BYTES`
/// ("hi") base64-encodes to "aGk=".
#[cfg(feature = "anthropic")]
#[tokio::test]
async fn a_recorded_request_carries_the_image_part_to_the_wire() {
    use tapir_provider::http::MockHttpClient;
    use tapir_provider::{AnthropicProvider, Credential};

    let mock = Arc::new(MockHttpClient::with_stream(vec![
        SAMPLE_STREAM.as_bytes().to_vec(),
    ]));
    let provider = AnthropicProvider::new(
        mock.clone(),
        Credential::api_key("sk-test"),
        "claude-fable-5",
    );
    let agent = Agent::builder().provider(provider).build().expect("build");

    let run = agent
        .prompt_with(UserInput::text("what is this?").with_image(png_image()))
        .expect("a hand-supplied provider never rejects an image");
    run.await.expect("run");

    let body: serde_json::Value =
        serde_json::from_slice(mock.last_request().body.as_deref().unwrap())
            .expect("the recorded request has a JSON body");
    let content = &body["messages"][0]["content"];
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[0]["text"], "what is this?");
    assert_eq!(content[1]["type"], "image");
    assert_eq!(content[1]["source"]["type"], "base64");
    assert_eq!(content[1]["source"]["media_type"], "image/png");
    assert_eq!(content[1]["source"]["data"], "aGk=");
}

/// The steered half of the same criterion: an image steered into a live
/// `converse` run reaches the *next* turn's HTTP request. Drives a real
/// `AnthropicProvider` over the recording transport across a turn boundary and
/// asserts the second recorded request carries the base64 image block.
#[cfg(feature = "anthropic")]
#[tokio::test]
async fn a_steered_image_reaches_the_next_wire_request() {
    use tapir_provider::http::MockHttpClient;
    use tapir_provider::{AnthropicProvider, Credential};

    // One stream per turn: turn 0 answers the opening prompt (parking the run),
    // turn 1 answers after the steered image lands.
    let mock = Arc::new(MockHttpClient::new());
    mock.push_stream(vec![SAMPLE_STREAM.as_bytes().to_vec()]);
    mock.push_stream(vec![SAMPLE_STREAM.as_bytes().to_vec()]);
    let provider = AnthropicProvider::new(
        mock.clone(),
        Credential::api_key("sk-test"),
        "claude-fable-5",
    );
    let agent = Agent::builder().provider(provider).build().expect("build");

    let mut events = agent.subscribe();
    let run = agent.converse("start");
    let handle = run.handle();
    let task = tokio::spawn(async move { run.await });

    wait_for_idle(&mut events).await;
    handle
        .steer_input(UserInput::text("here").with_image(png_image()))
        .expect("a hand-supplied provider never rejects an image steer");
    wait_for_idle(&mut events).await;
    handle.finish();
    tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .expect("run must complete")
        .expect("run task")
        .expect("run");

    let requests = mock.requests();
    assert_eq!(requests.len(), 2, "two turns produce two wire requests");
    let body: serde_json::Value =
        serde_json::from_slice(requests[1].body.as_deref().unwrap())
            .expect("the second recorded request has a JSON body");
    // History for turn 1 is [user "start", assistant reply, user "here"+image];
    // the steered image is the last message's second content part.
    let messages = body["messages"].as_array().expect("messages array");
    let content = &messages[messages.len() - 1]["content"];
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[0]["text"], "here");
    assert_eq!(content[1]["type"], "image");
    assert_eq!(content[1]["source"]["type"], "base64");
    assert_eq!(content[1]["source"]["media_type"], "image/png");
    assert_eq!(content[1]["source"]["data"], "aGk=");
}
