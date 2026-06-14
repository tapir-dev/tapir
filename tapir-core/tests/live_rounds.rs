//! The production-shaped provider seam, as integration tests: a `Runtime`
//! session whose turn is driven by `LiveRounds` over a *registered* provider, a
//! real HTTP round against a canned chat-shape SSE server. These live in
//! `tapir-core` (not `tapir-ai`) because the `Runtime`/`Session` they exercise
//! are the agent engine's, a layer above the LLM crate. The provider machinery
//! itself is reached through `tapir-core`'s re-exports of `tapir-ai`.

use tapir_core::agent::{Input, TurnEvent};
use tapir_core::providers::testing::{spawn_mock_sse, sse_text};
use tapir_core::providers::{
    Api, Creds, LiveRounds, Step, WireProvider, base_client, retrying_client,
};
use tapir_core::runtime::Runtime;

#[tokio::test]
async fn a_session_completes_a_turn_through_a_builtin_provider() {
    // The seam, production-shaped: a Runtime session whose turn is driven by
    // LiveRounds over the *registered* built-in deepseek Provider — a real
    // HTTP round against a canned chat-shape SSE server. No name-dispatch.
    let port = spawn_mock_sse(vec![sse_text("hello-from-wire")]);
    // SAFETY: only this test reads the deepseek base-URL override in-process
    // (the PTY test sets it on a child process).
    unsafe {
        std::env::set_var(
            "TAPIR_BASE_URL_DEEPSEEK",
            format!("http://127.0.0.1:{port}"),
        )
    };

    let rt = Runtime::builder().build();
    let provider =
        rt.find_provider("deepseek").expect("built-in deepseek is registered");
    let mut session = rt.session(".".into());
    session.submit(Input::text("hi"));

    let runner = LiveRounds {
        client: retrying_client(base_client(5), 0),
        provider,
        creds: Creds::ApiKey { key: "sk-test".into() },
        model: "deepseek-chat".into(),
        instructions: String::new(),
        tools: Vec::new(),
        effort: None,
    };
    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
    session.run_turn(&runner, &tx).await;
    drop(tx);

    let cap = std::time::Duration::from_secs(10);
    let (mut text, mut done) = (String::new(), false);
    while let Ok(Some(ev)) = tokio::time::timeout(cap, rx.recv()).await {
        match ev {
            TurnEvent::Text { delta } => text.push_str(&delta),
            TurnEvent::Done => done = true,
            _ => {}
        }
    }
    assert_eq!(
        text, "hello-from-wire",
        "the reply streamed through the provider trait"
    );
    assert!(done, "the turn completed");
    assert!(
        matches!(session.history().last(), Some(Step::Assistant { text, .. }) if text == "hello-from-wire"),
        "the assistant step landed in the session history",
    );

    unsafe { std::env::remove_var("TAPIR_BASE_URL_DEEPSEEK") };
}

#[tokio::test]
async fn a_custom_endpoint_provider_streams_with_no_new_streaming_code() {
    // The convenience case: a new OpenAI-compatible provider is just an
    // endpoint, credentials, and a model list over the built-in Chat wire
    // shape — registered like any Provider, with no new streaming code.
    let port = spawn_mock_sse(vec![sse_text("custom-endpoint-reply")]);
    let myai = WireProvider::new("myai", Api::Chat)
        .endpoint(format!("http://127.0.0.1:{port}"))
        .api_key("sk-custom")
        .models(["my-model-1"]);
    let rt = Runtime::builder().provider(std::sync::Arc::new(myai)).build();
    let p = rt.find_provider("myai").expect("the custom provider registered");

    // It advertises the supplied model list (the catalog has no "myai")…
    let ids: Vec<String> = p.models().iter().map(|m| m.id.clone()).collect();
    assert_eq!(ids, ["my-model-1"], "the supplied models are advertised");

    // …resolves credentials to the explicit key (no env var, no auth.toml)…
    let creds = p
        .creds(&base_client(5), rt.credentials().as_ref())
        .await
        .expect("creds resolve");
    assert!(
        matches!(creds, Creds::ApiKey { ref key } if key == "sk-custom"),
        "the explicit key is the credential",
    );

    // …and a session turn streams through the shared chat-shape code.
    let mut session = rt.session(".".into());
    session.submit(Input::text("hi"));
    let runner = LiveRounds {
        client: retrying_client(base_client(5), 0),
        provider: p,
        creds,
        model: "my-model-1".into(),
        instructions: String::new(),
        tools: Vec::new(),
        effort: None,
    };
    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
    session.run_turn(&runner, &tx).await;
    drop(tx);

    let cap = std::time::Duration::from_secs(10);
    let mut text = String::new();
    while let Ok(Some(ev)) = tokio::time::timeout(cap, rx.recv()).await {
        if let TurnEvent::Text { delta } = ev {
            text.push_str(&delta);
        }
    }
    assert_eq!(
        text, "custom-endpoint-reply",
        "the custom endpoint served the round"
    );
}
