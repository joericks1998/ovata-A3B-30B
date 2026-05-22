"""
Teacher model for distillation dataset generation.

Calls Claude Haiku 4.5 with tool_use to produce short trajectories (1-3 rounds),
converts the output to v0.2 template format, and streams records to a writer.

Output format per record:
  {
    "messages": [
      {"role": "user",      "content": "<task>"},
      {"role": "assistant", "content": "<call_tool: \"name\", arg0, arg1>"},
      {"role": "tool",      "content": "<result>"},
      ...
      {"role": "assistant", "content": "<terminal or next call>"},
    ]
  }

Assistant turns store pre-rendered <call_tool:...> strings directly in content.
The chat template renders them verbatim via {{ message.content | trim }}.
"""

import asyncio
import json
import random
from collections.abc import Callable
from typing import Any

import anthropic

HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"

_SYSTEM_PREFIX = (
    "You are an execution engine. "
    "Use the available tools to complete every task. "
    "Do not explain or chat.\n\n"
    "Available tools:\n"
)

# Plausible stub results keyed by tool name.
# Variety matters: stops the model from overfitting to a single result per tool.
_STUBS: dict[str, list[str]] = {
    "read_file": [
        "# module\ndef run(ctx):\n    return ctx.input",
        "name,age\nalice,30\nbob,25\ncarol,40",
        '{"version": "1.2", "debug": false, "workers": 4}',
        "hello world",
        "line1\nline2\nline3",
    ],
    "write_file":   ["ok"],
    "delete_file":  ["ok"],
    "list_dir": [
        '["main.jde", "utils.jde", "tests/"]',
        '["config.json", "data.csv", "README.md"]',
        '["a.jde", "b.jde"]',
    ],
    "shell": [
        '{"stdout": "total 12\\n-rw-r--r-- 1 user user 256 main.jde\\n", "stderr": "", "exit_code": 0}',
        '{"stdout": "ok\\n", "stderr": "", "exit_code": 0}',
        '{"stdout": "", "stderr": "", "exit_code": 0}',
    ],
    "http_get": [
        '{"status": 200, "body": {"ok": true, "items": [1, 2, 3]}}',
        '{"status": 200, "body": {"value": 42}}',
        '{"status": 404, "body": {"error": "not found"}}',
    ],
    "http_post": [
        '{"status": 200, "body": {"id": "abc123", "created": true}}',
        '{"status": 201, "body": {"id": "xyz789"}}',
    ],
    "memory_get":    ['"cached_value"', "42", '{"key": "val"}', "null"],
    "memory_set":    ["ok"],
    "memory_delete": ["ok"],
}


def _stub(tool_name: str) -> str:
    return random.choice(_STUBS.get(tool_name, ['"ok"']))


def _args_positional(name: str, arguments: dict, tools_by_name: dict) -> list:
    """Order Claude's dict arguments by the tool schema's parameter declaration order."""
    props = tools_by_name.get(name, {}).get("parameters", {}).get("properties", {})
    required = tools_by_name.get(name, {}).get("parameters", {}).get("required", [])
    # required first, then any optional props, in declaration order
    ordered = required + [k for k in props if k not in required]
    return [arguments[k] for k in ordered if k in arguments]


def _render_call(name: str, pos_args: list) -> str:
    if pos_args:
        args_str = ", ".join(json.dumps(a) for a in pos_args)
        return f'<call_tool: "{name}", {args_str}>'
    return f'<call_tool: "{name}">'


def _build_system(tools: list[dict]) -> list[dict]:
    """Cached system block — tool manifest is large, cache saves cost at scale."""
    return [
        {
            "type": "text",
            "text": _SYSTEM_PREFIX + json.dumps(tools, indent=2),
            "cache_control": {"type": "ephemeral"},
        }
    ]


def _anthropic_tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
        }
        for t in tools
    ]


class Teacher:
    """
    Async teacher that generates tool-call trajectories via Claude Haiku 4.5.

    Usage:
        teacher = Teacher(tools)
        await teacher.run(tasks, on_record=lambda r: out.write(json.dumps(r) + "\\n"))
    """

    def __init__(
        self,
        tools: list[dict],
        model: str = HAIKU,
        max_tokens: int = 512,
        max_rounds: int = 3,
        concurrency: int = 30,
    ) -> None:
        self._client = anthropic.AsyncAnthropic()
        self._tools = tools
        self._tools_by_name = {t["name"]: t for t in tools}
        self._anthropic_tools = _anthropic_tools(tools)
        self._system = _build_system(tools)
        self.model = model
        self.max_tokens = max_tokens
        self.max_rounds = max_rounds
        self._sem = asyncio.Semaphore(concurrency)

    async def generate(self, task: str) -> dict:
        """Generate one training record for a task."""
        async with self._sem:
            turns = await self._trajectory(task)
        return {"messages": [{"role": "user", "content": task}, *turns]}

    async def run(
        self,
        tasks: list[str],
        on_record: Callable[[dict], None],
        log_every: int = 200,
    ) -> tuple[int, int]:
        """
        Generate trajectories for all tasks, calling on_record for each success.
        Returns (written, errors).
        """
        written = 0
        errors = 0
        lock = asyncio.Lock()

        async def _one(i: int, task: str) -> None:
            nonlocal written, errors
            try:
                record = await self.generate(task)
                async with lock:
                    on_record(record)
                    written += 1
                    if written % log_every == 0:
                        print(f"  [{written}/{len(tasks)}] errors={errors}")
            except Exception as exc:
                async with lock:
                    errors += 1
                print(f"  ERROR [{i}]: {exc!r} | {task[:60]}")

        await asyncio.gather(*[_one(i, t) for i, t in enumerate(tasks)])
        return written, errors

    # ── internals ──────────────────────────────────────────────────────────────

    async def _trajectory(self, task: str) -> list[dict[str, Any]]:
        """Run one multi-turn tool loop and return the turn list."""
        # Messages sent to Claude (includes raw content blocks for tool_use continuations)
        api_messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
        # Clean turn dicts for the training record
        trajectory: list[dict] = []

        for _ in range(self.max_rounds):
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self._system,
                messages=api_messages,
                tools=self._anthropic_tools,
            )

            calls = [b for b in response.content if b.type == "tool_use"]
            texts = [b for b in response.content if b.type == "text"]

            if calls:
                # Render all calls as <call_tool:...> lines in one assistant turn
                rendered = "\n".join(
                    _render_call(
                        c.name,
                        _args_positional(c.name, c.input, self._tools_by_name),
                    )
                    for c in calls
                )
                trajectory.append({"role": "assistant", "content": rendered})

                # One tool result block per call (spec: one block per result)
                tool_results_api = []
                for call in calls:
                    result = _stub(call.name)
                    trajectory.append({"role": "tool", "content": result})
                    tool_results_api.append({
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": result,
                    })

                # Advance the API conversation
                api_messages.append({"role": "assistant", "content": response.content})
                api_messages.append({"role": "user", "content": tool_results_api})

            else:
                # Terminal text response
                text = texts[0].text.strip() if texts else ""
                trajectory.append({"role": "assistant", "content": text})
                break

            if response.stop_reason != "tool_use":
                break

        return trajectory
