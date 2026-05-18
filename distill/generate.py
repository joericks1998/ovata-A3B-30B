"""
Distillation dataset generator.

Uses Claude as teacher to produce (task → tool_call_trajectory) training pairs
formatted with the Ovata chat template. Stubs tool execution during generation
— real execution would require a live Jade runtime.

Usage:
    python distill/generate.py --config distill/config.yaml
    python distill/generate.py --config distill/config.yaml --output data/raw/samples.jsonl
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Any

import anthropic

_CACHE = {"type": "ephemeral"}

EXECUTOR_SYSTEM = (
    "You are an execution engine. Use the available tools to complete every task.\n"
    "Do not explain or chat. Every response must consist solely of tool calls.\n\n"
    "Tool call format — one per line:\n"
    '<|tool_call|>{{"name": "tool_name", "arguments": {{"param": "value"}}}}<|tool_call|>\n\n'
    "Available tools:\n{tool_manifest}"
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_tools(schemas_dir: Path) -> list[dict]:
    tools: list[dict] = []
    for p in sorted(schemas_dir.glob("*.json")):
        with open(p) as f:
            data = json.load(f)
            tools.extend(data if isinstance(data, list) else [data])
    return tools


def tools_to_anthropic_format(tools: list[dict]) -> list[dict]:
    return [
        {
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
        }
        for t in tools
    ]


def stub_tool_result(tool_name: str, tool_input: dict) -> str:
    return json.dumps({"name": tool_name, "result": f"__stub__{tool_name}"})


def generate_trajectory(
    client: anthropic.Anthropic,
    task: str,
    tools: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
    max_rounds: int = 10,
) -> list[dict[str, Any]]:
    system_text = EXECUTOR_SYSTEM.format(tool_manifest=json.dumps(tools, indent=2))
    messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
    trajectory: list[dict[str, Any]] = []

    for _ in range(max_rounds):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=[{"type": "text", "text": system_text, **_CACHE}],
            messages=messages,
            tools=tools_to_anthropic_format(tools),
        )

        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})
        trajectory.append({"role": "assistant", "content": assistant_content})

        if response.stop_reason != "tool_use":
            break

        tool_results = [
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": stub_tool_result(block.name, block.input),
            }
            for block in assistant_content
            if block.type == "tool_use"
        ]

        messages.append({"role": "user", "content": tool_results})
        trajectory.append({"role": "tool", "content": tool_results})

    return trajectory


def trajectory_to_training_record(task: str, trajectory: list[dict]) -> dict:
    return {
        "messages": [{"role": "user", "content": task}, *trajectory],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="data/raw/samples.jsonl")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tools = load_tools(Path("distill/schemas"))
    client = anthropic.Anthropic()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tasks: list[str] = cfg["tasks"]
    n_samples: int = cfg.get("n_samples", len(tasks))
    model: str = cfg.get("teacher_model", "claude-opus-4-7")
    max_tokens: int = cfg.get("max_tokens", 2048)
    temperature: float = cfg.get("temperature", 0.7)

    written = 0
    errors = 0
    with open(output_path, "w") as out:
        for i in range(n_samples):
            task = tasks[i % len(tasks)]
            try:
                traj = generate_trajectory(client, task, tools, model, max_tokens, temperature)
                record = trajectory_to_training_record(task, traj)
                out.write(json.dumps(record) + "\n")
                written += 1
                print(f"[{written}/{n_samples}] {task[:72]}")
            except Exception as exc:
                errors += 1
                print(f"  ERROR ({errors}): {exc}")

    print(f"\nDone — {written} samples written to {output_path} ({errors} errors)")


if __name__ == "__main__":
    main()
