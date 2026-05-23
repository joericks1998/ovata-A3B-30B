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
| `<\|system\|>` | Open system block | Jade (pre-filled) |
| `<\|user\|>` | Open user block | Jade (pre-filled) |
| `<\|model\|>` | Open model block | Jade (generation prompt) |
| `<\|tool_response\|>` | Open tool result block | Jade (pre-filled) |
| `<\|memory\|>` | Open memory block | Jade (pre-filled) |
| `<\|call_tool:` | Tool call prefix | Model |

### Already in Qwen3 vocab

| Token | Role | Who writes it |
|---|---|---|
| `<\|im_end\|>` | Universal block closer | Both |
| `<think>` | Open reasoning block | Model |
| `</think>` | Close reasoning block | Model |

---

## Role Blocks

### `<|system|>`
Injected once at context start. Contains:
1. The executor persona (one sentence)
2. The tool manifest as a JSON array

Jade controls what tools are visible per-call by varying the manifest.

### `<|user|>`
The task or instruction to execute. Typically a structured spec in Jade's loop, but the model is trained to handle natural language too.

### `<|model|>`
The model's response. Structure:

```
<|model|>
[<think>
optional reasoning
</think>]
<|call_tool: "name", arg0|> | terminal_value
<|im_end|>
```

- The optional `<think>...</think>` block precedes any tool calls or terminal value
- Thinking is **inference-only** — training data contains no `<think>` blocks
- After thinking (if any): one or more tool calls **or** a terminal result string — never both in the same turn

### `<|tool_response|>`
A tool result injected by Jade after executing a call. One block per call result.

### `<|memory|>`
Persistent state injected by Jade at any point in context. The model reads it but never writes it. Jade controls when and what is injected — e.g. retrieved memories, prior task summaries, or scratchpad state. Multiple blocks are allowed per context.

---

## Tool Call Format

```
<|call_tool: "tool_name", arg0, arg1, ...|>
```

Rules:
- `<|call_tool:` is a single special token
- Tool name is a quoted string immediately after the token
- Arguments are positional, comma-separated, each rendered as JSON scalar values
- `|>` self-delimits the call
- Multiple calls in one turn: one per line
- Jade's compile-time GBNF grammar constrains tool name to the manifest and args to each tool's schema

---

## Multi-Turn Tool Loop Example

```
<|system|>
You are an execution engine. Use the available tools to complete every task.

Available tools:
[{"name": "read_file", ...}, {"name": "write_file", ...}]
<|im_end|>
<|user|>
Copy /src/a.txt to /dst/b.txt
<|im_end|>
<|model|>
<|call_tool: "read_file", "/src/a.txt"|>
<|im_end|>
<|tool_response|>
hello world
<|im_end|>
<|model|>
<|call_tool: "write_file", "/dst/b.txt", "hello world"|>
<|im_end|>
<|tool_response|>
ok
<|im_end|>
<|model|>
done
<|im_end|>
```

---

## Open Questions

- [ ] Parallel tool calls in one turn: multiple lines (current) vs. single grammar rule?
- [x] `<think>` blocks: supported at inference, excluded from training data — thinking emerges from Qwen3-Base priors
- [ ] Tool error format in `<|tool_response|>` — plain string or structured?
