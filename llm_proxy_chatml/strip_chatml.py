#!/usr/bin/env python3
import json, sys

with open(sys.argv[1]) as f:
    data = json.load(f)

keep = {"role", "content", "tool_calls", "tool_call_id", "name", "reasoning_content", "reasoning"}
out = []
for m in data.get("messages", []):
    if m.get("role") == "system":
        continue
    out.append({k: v for k, v in m.items() if k in keep and v is not None})

print(json.dumps(out, ensure_ascii=False, indent=2))
