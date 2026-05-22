# Ovata Chat Template Spec — v0.2

## Design Principles

- The model is a **blind executor** — it has no awareness of Jade lang
- Everything the model can do is expressed as a tool
- The template is a contract between Jade's runtime and the model
- Jade's compile-time GBNF grammar enforces structural correctness; the template provides clean seams

---

## Special Tokens

### New tokens (must be added to Qwen3-Base vocab)

| Token | Role | Who writes it |
|---|---|---|
| `<system>` | Open system block | Jade (pre-filled) |
| `<user>` | Open user block | Jade (pre-filled) |
| `<model>` | Open model block | Jade (generation prompt) |
| `<tool_response>` | Open tool result block | Jade (pre-filled) |
| `<call_tool:` | Tool call prefix | Model |

### Already in Qwen3 vocab

| Token | Role | Who writes it |
|---|---|---|
| `<|im_end|>` | Universal block closer | Both |

---

## Role Blocks

### `<system>`
Injected once at context start. Contains:
1. The executor persona (one sentence)
2. The tool manifest as a JSON array

Jade controls what tools are visible per-call by varying the manifest.

### `<user>`
The task or instruction to execute. Typically a structured spec in Jade's loop, but the model is trained to handle natural language too.

### `<model>`
The model's response. Must be **either**:
- One or more tool calls — the primary case
- A terminal result string — only when a task produces a direct scalar value

No mixing of tool calls and freeform text in the same turn.

### `<tool_response>`
A tool result injected by Jade after executing a call. One block per call result.

---

## Tool Call Format

```
<call_tool: "tool_name", arg0, arg1, ...>
```

Rules:
- `<call_tool:` is a single special token
- Tool name is a quoted string immediately after the token
- Arguments are positional, comma-separated, each rendered as JSON scalar values
- No closing tag — the `>` self-delimits the call
- Multiple calls in one turn: one per line
- Jade's compile-time GBNF grammar constrains tool name to the manifest and args to each tool's schema

---

## Multi-Turn Tool Loop Example

```
<system>
You are an execution engine. Use the available tools to complete every task.

Available tools:
[{"name": "read_file", ...}, {"name": "write_file", ...}]
<|im_end|>
<user>
Copy /src/a.txt to /dst/b.txt
<|im_end|>
<model>
<call_tool: "read_file", "/src/a.txt">
<|im_end|>
<tool_response>
hello world
<|im_end|>
<model>
<call_tool: "write_file", "/dst/b.txt", "hello world">
<|im_end|>
<tool_response>
ok
<|im_end|>
<model>
done
<|im_end|>
```

---

## Open Questions

- [ ] Parallel tool calls in one turn: multiple lines (current) vs. single grammar rule?
- [ ] Should `<think>` blocks be suppressed entirely for a blind executor, or excluded from training data?
- [ ] Tool error format in `<tool_response>` — plain string or structured?
