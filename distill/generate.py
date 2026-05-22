"""
Distillation dataset generator.

Drives the Teacher to produce (task → tool_call_trajectory) JSONL training data.
Records are written incrementally — safe to kill and resume (append mode).

Usage:
    python distill/generate.py --config distill/config.yaml
    python distill/generate.py --config distill/config.yaml --output data/raw/samples.jsonl --append
"""

import argparse
import asyncio
import json
import yaml
from pathlib import Path

from distill.teacher import Teacher, HAIKU


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


def expand_tasks(cfg: dict) -> list[str]:
    """Repeat and shuffle task list to reach n_samples."""
    import random
    base: list[str] = cfg["tasks"]
    n: int = cfg.get("n_samples", len(base))
    repeated = (base * ((n // len(base)) + 1))[:n]
    random.shuffle(repeated)
    return repeated


async def main_async(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    tools = load_tools(Path("distill/schemas"))
    tasks = expand_tasks(cfg)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"

    teacher = Teacher(
        tools=tools,
        model=cfg.get("teacher_model", HAIKU),
        max_tokens=cfg.get("max_tokens", 512),
        max_rounds=cfg.get("max_rounds", 3),
        concurrency=cfg.get("concurrency", 30),
    )

    print(f"Generating {len(tasks)} samples → {output_path} (mode={mode})")
    print(f"Model: {teacher.model}  rounds≤{teacher.max_rounds}  concurrency={teacher._sem._value}")

    with open(output_path, mode) as out:
        def write(record: dict) -> None:
            out.write(json.dumps(record) + "\n")
            out.flush()

        written, errors = await teacher.run(tasks, on_record=write)

    print(f"\nDone — {written} written, {errors} errors → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="data/raw/samples.jsonl")
    parser.add_argument("--append", action="store_true", help="Append to existing output file")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
