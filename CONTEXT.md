# tapir

A stateful LLM agent SDK built on `tapir-provider`. This glossary fixes the
vocabulary of the agent core: the engine, its state, and the persistence seam.

## Language

**Agent**:
The stateful driver that owns a conversation and runs the tool loop over a
`Provider`. Generic over a custom-message type `M`.

**AgentMessage**:
The unit of conversation state the `Agent` owns: either a provider `Message`
(bridged to the LLM) or an app-defined `Custom(M)`. External-tagged serde is its
on-disk form.
_Avoid_: entry, record, transcript line

**SessionStore**:
The persistence seam the `Agent` writes through: an object-safe, async trait
that appends `AgentMessage`s and loads them back. One store instance is bound to
one session. Concrete backends are feature-gated; the core only defines the seam.
_Avoid_: persister, repository, session backend (a backend _implements_ a SessionStore)

**Ephemeral**:
An `Agent` with no `SessionStore` (the default). Its state lives only in memory
and is never persisted.
_Avoid_: stateless, in-memory-only, transient

**Write-through**:
The discipline of persisting each `AgentMessage` to the `SessionStore` at the
moment it is appended to state: the user message durably before the provider
call, assistant and tool artifacts after they settle. Durable-on-await: an
`append` future resolves only once the message is persisted.
_Avoid_: autosave, snapshot, flush

**Resume**:
Reconstructing an `Agent` from a `SessionStore` by loading its persisted history
as the initial state. The async counterpart to constructing an ephemeral Agent.
_Avoid_: rehydrate, restore, reopen
