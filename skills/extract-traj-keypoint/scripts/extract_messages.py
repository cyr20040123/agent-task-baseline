#!/usr/bin/env python3
"""Print ChatML trajectory messages, stripping system messages and noise fields."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_EXCLUDE_KEYS = {"timestamp", "prompt_ids", "completion_ids", "logprobs"}


def filter_message(msg: dict) -> dict:
    return {k: v for k, v in msg.items() if k not in _EXCLUDE_KEYS}


def extract(file: Path) -> list[dict]:
    data = json.loads(file.read_text(encoding="utf-8"))
    messages = data.get("messages", data) if isinstance(data, dict) else data
    return [filter_message(m) for m in messages if m.get("role") != "system"]


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Print ChatML trajectory messages, excluding system messages and noise fields."
    )
    parser.add_argument(
        "file",
        type=Path,
        help="Path to the ChatML JSON trajectory file",
    )
    args = parser.parse_args()

    if not args.file.exists():
        print(f"File not found: {args.file}", file=sys.stderr)
        return 1

    messages = extract(args.file)
    json.dump(messages, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
