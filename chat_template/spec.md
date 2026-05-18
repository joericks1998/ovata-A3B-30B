# Ovata Chat Template Spec — v0.1

## Design Principles

- The model is a **blind executor** — it has no awareness of Jade lang
- Everything the model can do is expressed as a tool
- The template is a contract between Jade's runtime and the model
- Grammar enforcement (GBNF) handles structural correctness; the template just provides clean seams

---

## Special Tokens

| Token | Role |
|---|---|
| `<\|im_start\|>` | Open a role block |
| `<\|im_end\|>` | Close a role block |
| `<\|tool_call\|>` | Wraps a tool call JSON body (open **and** close delimiter) |

`<\|tool_call\|>` must be added to the tokenizer vocabulary as an additional special token before fine-tuning.

---

## Role Blocks

### `system`
Injected once at context start. Contains:
1. The executor persona (one sentence)
2. The tool manifest as a JSON array

Jade controls what tools are visible per-call by varying the manifest.

### `user`
The task or instruction to execute. In Jade's loop this is typically a structured spec, not natural language — but the model is trained to handle both.

### `assistant`
The model's response. Must be **either**:
- One or more tool calls — the primary case
- A terminal result string — only when a task produces a direct scalar value

No mixing of tool calls and freeform text in the same turn.

### `tool`
A tool result injected by Jade after executing a call. One block per call result.
Format: `{"name": "tool_name", "result": <value>}`
Errors use: `{"name": "tool_name", "error": "<message>"}`

---

## Tool Call Format

```
<|tool_call|>{"name": "tool_name", "arguments": {"param": "value"}}<|tool_call|>
```

Rules:
- Content is a JSON object with exactly two keys: `name` (string) and `arguments` (object)
- Multiple calls in one turn: one per line, each wrapped independently
- No trailing comma, no whitespace outside the JSON object
- GBNF grammar in `grammar.gbnf` enforces this at inference time

---

## Multi-Turn Tool Loop Example

```
<|im_start|>system
You are an execution engine. Use the available tools to complete every task.

Available tools:
[{"name": "read_file", ...}, {"name": "write_file", ...}]
<|im_end|>
<|im_start|>user
Copy /src/a.txt to /dst/b.txt
<|im_end|>
<|im_start|>assistant
<|tool_call|>{"name": "read_file", "arguments": {"path": "/src/a.txt"}}<|tool_call|>
<|im_end|>
<|im_start|>tool
{"name": "read_file", "result": "hello world"}
<|im_end|>
<|im_start|>assistant
<|tool_call|>{"name": "write_file", "arguments": {"path": "/dst/b.txt", "content": "hello world"}}<|tool_call|>
<|im_end|>
<|im_start|>tool
{"name": "write_file", "result": "ok"}
<|im_end|>
<|im_start|>assistant
done
<|im_end|>
```

---

## Open Questions

- [ ] Parallel tool calls in one turn: multiple lines (current) vs. single JSON array?
- [ ] Should Qwen3's `<think>` blocks be allowed, suppressed, or controlled per-call via system prompt?
- [ ] Max nesting depth for grammar argument enforcement?
- [ ] Tool error retry behavior: does Jade re-inject the turn or pass error to model?
