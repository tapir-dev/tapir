# tapir — Domain Glossary

`tapir` is a stateful LLM agent SDK built on the sibling `tapir-provider` crate.
This file is a glossary only: canonical terms and what they mean, no implementation
detail. Terms owned by `tapir-provider` (`Provider`, `Context`, `Message`,
`ContentPart`, `AssistantMessage`) are reused as-is and not redefined here.

## Terms

- **Agent** — The stateful object owning the in-memory conversation history and
  configuration; it drives runs. Generic over a custom-message type, defaulting to
  `NoCustom` (the zero-ceremony common case).

- **Run** — One invocation of the agent (`prompt` / `resume` / `converse`), modeled
  as both a stream of `AgentEvent` and a future resolving to the final reply. A run
  executes on its own task.

- **Turn** — One iteration of a run's loop: a single provider completion plus the
  batch of tool calls it requests.

- **AgentMessage** — The SDK's message supertype layered over provider `Message`s. An
  `AgentMessage` may be a custom or UI-only message that never reaches the model.

- **CustomMessage** — A caller-defined message type carried in history. It either
  converts to a provider `Message` or stays UI-only.

- **Tool** — A typed capability the agent can invoke. It has typed arguments and a
  concurrency class.

- **Concurrency class** — `Safe` (parallelizable reads) vs `Exclusive` (serialized
  mutations); governs how a batch of tool calls is executed.

- **AgentEvent** — The single flat event type streamed from a run.

- **Session** — One conversation as persisted through a SessionStore.

- **SessionStore** — The persistence seam: append-one plus load of `AgentMessage`s.
  Absent, the agent is ephemeral.

- **SchemaProfile** — Per-provider normalization of a tool's JSON schema on the send
  path.

- **Steering** — Injecting input into a live run, drained at a turn boundary.

- **Follow-up** — A run that parks at a tool-free reply (an `Idle` state) awaiting
  more input instead of ending.

- **RunHandle** — The cloneable control surface (abort / steer / finish) that outlives
  the `Run`.
