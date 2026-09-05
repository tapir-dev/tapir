# tapir

A stateful LLM agent SDK built on `tapir-provider`: a run loop, a typed tool
system, an event stream, and in-memory conversation state with a persistence
seam.

## Language

### Tool scheduling

**Concurrency class**:
A tool's declared scheduling class, `Safe` or `Exclusive`, that tells the
scheduler what may run alongside it. Defaults to `Exclusive` (assume mutation),
so an unmarked tool is never parallelized.

**Safe**:
A tool that only reads and may run concurrently with other `Safe` tools. Read
semantics of a read/write lock: many `Safe` tools share the batch, but none
overlaps an `Exclusive` one.
_Avoid_: read-only (it is derived from `Safe`, not a separate axis), parallel

**Exclusive**:
A tool that may mutate state and runs alone: no other tool, `Safe` or
`Exclusive`, overlaps it. Write semantics of a read/write lock — a true
barrier, not merely serialized against other `Exclusive` tools.
_Avoid_: serial, mutating, locked

**Tool batch**:
The set of tool calls a single assistant turn requests, executed together
before the next turn. The scheduler runs the batch under the concurrency-class
rules and appends one result per call in model-requested order.
_Avoid_: tool round, tool group
