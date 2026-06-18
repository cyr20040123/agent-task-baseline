# llm_proxy_chatml

OpenAI-compatible LLM proxy with ChatML conversation logging. Forwards requests from clients to an upstream LLM service, records conversation history with per-message timestamps, and supports both streaming and non-streaming requests.

## Features

- **Transparent proxying** — `/v1/chat/completions`, `/v1/completions`, and catch-all forwarding for any other endpoint (e.g. `/v1/models`)
- **Streaming & non-streaming** — SSE stream reconstruction and passthrough
- **ChatML session logging** — Multi-turn conversation tracking with prefix matching, output as JSON with per-message ISO timestamps
- **Persistent config** — Parameters auto-saved to `llm_proxy.yaml`, with preset groups and diff-based RECENT tracking
- **Graceful shutdown** — Dumps pending ChatML sessions on SIGINT/SIGTERM

## Installation

```bash
pip install fastapi uvicorn httpx pyyaml
```

## Quick Start

```bash
python llm_proxy.py --base-url http://localhost:8000 --api-key sk-your-key --log-chatml
```

This starts the proxy on `0.0.0.0:8030`, forwarding to `http://localhost:8000` with ChatML logging enabled.

Then point your OpenAI client at `http://localhost:8030/v1`.

## Configuration

All parameters can be set via command line, `llm_proxy.yaml`, or both. The config file uses a YAML structure with named groups.

**Resolution priority (highest to lowest):**

```
1. CLI arguments
2. --preset group        (if --preset NAME is specified)
3. RECENT group          (auto-managed, only stores diffs from DEFAULT)
4. DEFAULT group         (baseline configuration)
5. Hardcoded defaults
```

On startup, the proxy logs all resolved parameters; values differing from `DEFAULT` are marked with `*`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--host` | `0.0.0.0` | Proxy listen address |
| `--port` | `8030` | Proxy listen port |
| `--base-url` | *(required)* | Upstream LLM service URL (e.g. `http://localhost:8000`). The `/v1` prefix is auto-detected on startup if missing. |
| `--api-key` | `""` | Upstream API key — replaces any key sent by the client |
| `--log-folder` | `./logs/` | Directory for logs and ChatML output |
| `--log-chatml` | `none` | ChatML recording mode: `none` (disabled), `multi` (prefix-matched multi-turn), `single` (one entry per request) |
| `--session-name` | `sess_MMdd_HHmmss` | Name for the initial ChatML session |
| `--session-path` | *(--log-folder)* | ChatML output directory (defaults to `--log-folder`) |
| `--temperature` | `-1.0` | Default temperature injected into upstream requests when absent from the client request. Disabled when negative |
| `--rl` | `false` | Enable RL-specific ChatML logging — requests logprobs & token_ids from upstream, records them alongside each assistant response |
| `--default-model` | `None` | Default model name when the request's `model` field is empty or `"none"`. If not specified and upstream is reachable, auto-populated from the first available model. |
| `--preset` | `None` | Name of a YAML config group to load (e.g. `DEEPSEEK`). Values from this group override RECENT and DEFAULT. |

The config file is `llm_proxy.yaml` in the working directory. See the sample file for the structure.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/proxyhealth` | Health check — returns `{"status": "ok"}` |
| POST | `/newsession` | Dump current ChatML sessions to file, switch to a new session. Body: `{"session_name": "...", "session_path": "..."}` |
| GET | `/session_chats` | Return all current sessions in ChatML JSON format (read-only, does not dump to file). Requires `--log-chatml` ≠ `none` |
| POST | `/change_override_model` | Override the `model` field in forwarded requests, or clear the override. Body: `{"model": "model-id"}` |
| POST | `/v1/chat/completions` | Forward to upstream, record in ChatML session |
| POST | `/v1/completions` | Legacy completions endpoint |
| * | `/{path}` | Catch-all — forwards any other request to upstream |

## Model Handling

Model resolution follows a three-tier priority:

```
1. model_override   (`POST /change_override_model` — forces all requests)
   ↓  not set
2. default_model    (--default-model or auto-detected — fills empty / "none")
   ↓  not set or model already valid
3. pass-through      (client's model sent as-is)
```

### `/change_override_model` — Force Override

Sets a global model override applied to **every** request, regardless of what the client sends.

```
POST /change_override_model  {"model": "Qwen/Qwen3.5-4B"}
                              → override = "Qwen/Qwen3.5-4B"
client sends  "model": "gpt-4"  →  "model": "Qwen/Qwen3.5-4B"
client sends  no model field    →  "model": "Qwen/Qwen3.5-4B"
```

```
POST /change_override_model  {"model": ""}  or  {"model": "none"}
                              → override = None (cleared, back to pass-through)
client sends  "model": "gpt-4"  →  "model": "gpt-4"
```

### `--default-model` / Auto-detect — Fallback

Fills in the model name **only when** the client request has an empty (`""`) or `"none"` model, or omits the field entirely. Does **not** override a valid model name.

```
--default-model not set + upstream reachable
                         → default_model = first model from GET /v1/models

--default-model Qwen/Qwen3.5-4B
                         → default_model = "Qwen/Qwen3.5-4B"

client sends  "model": ""       →  "model": default_model
client sends  "model": "none"   →  "model": default_model
client sends  no model field    →  "model": default_model
client sends  "model": "gpt-4"  →  "model": "gpt-4"  (preserved)
```

- Both mechanisms apply to `/v1/chat/completions` and `/v1/completions`.
- On startup, the proxy fetches the available model list from upstream `GET /v1/models` and stores it in `app.state.available_models`.

## ChatML Output

When `--log-chatml` is set to `multi` or `single`, conversation history is saved as JSON files in `--session-path` (defaults to `--log-folder`).

### `multi` mode (prefix-matched)

Conversations are grouped by prefix matching — each incoming request's message history is matched against tracked sessions. Output filename: `{session_name}.chatml.json` (or `{session_name}_{i}.chatml.json` when multiple sessions exist).

```json
{
  "messages": [
    {"role": "system", "content": "You are helpful.", "timestamp": ""},
    {"role": "user", "content": "Hello", "timestamp": "2026-05-19T06:04:22.669170+00:00"},
    {"role": "assistant", "content": "Hi there!", "timestamp": "2026-05-19T06:04:22.685971+00:00"}
  ],
  "remarks": {"incomplete": false}
}
```

### `single` mode

Each request/response pair is stored as a separate entry. All entries are written to one file: `{session_name}.json`. Timestamps are omitted.

```json
[
  {"messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
  ]}
]
```

### RL fields (when `--rl` is enabled)

With `--rl`, each assistant response message gains three additional fields at the same level as `timestamp`:

| Field | Source | Description |
|-------|--------|-------------|
| `prompt_ids` | `prompt_token_ids` | Token IDs of the entire prompt |
| `completion_ids` | `choices[*].token_ids` | Token IDs generated for this response |
| `logprobs` | `choices[*].logprobs.content[*].logprob` | Log-probability for each generated token |

Example assistant message with RL fields:

```json
{
  "role": "assistant",
  "content": "红色",
  "timestamp": "2026-06-04T07:42:48.495115+00:00",
  "prompt_ids": [248045, 846, 198, ...],
  "completion_ids": [101698, 248046],
  "logprobs": [-0.31381985545158386, -0.0022499265614897013]
}
```

Internally, the proxy injects `"logprobs": true` and `"return_token_ids": true` into upstream requests when `--rl` is active. Fields are extracted from the upstream response and stored in the ChatML output.

### General notes

- Messages use standard OpenAI roles (`system`, `user`, `assistant`, `tool`)
- Each message has an ISO 8601 `timestamp` — empty string for messages the proxy didn't directly witness (`multi` mode) or always empty (`single` mode)
- `remarks.incomplete` is `true` when a conversation was cut off mid-turn (the last message is not from the assistant)
- Tool calls are stored in the `tool_calls` field of assistant messages
- Tool definitions are included in the top-level `tools` array when present
- Sessions are dumped when `/newsession` is called or on shutdown

### How session matching works

Each incoming request carries the full message history. The proxy matches it against tracked sessions by prefix — if a session's messages form a prefix of the request's messages, the new messages are appended to that session. This naturally groups requests into multi-turn conversations without requiring session IDs.

## Testing

```bash
# Run against the built-in mock upstream (no external dependencies)
python test_proxy.py

# Run against a real upstream
python test_proxy.py --real-upstream http://localhost:8030

# Test an already-running proxy
python test_proxy.py --proxy-url http://localhost:8031
```

## Files

| File | Purpose |
|------|---------|
| `llm_proxy.py` | Entry point — config, logging, server startup |
| `proxy_server.py` | FastAPI app with all route handlers and upstream forwarding |
| `chatml_session.py` | Session manager — prefix matching, ChatML JSON output |
| `test_proxy.py` | Integration tests with mock upstream |
