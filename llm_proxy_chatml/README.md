# llm_proxy_chatml

OpenAI-compatible LLM proxy with ChatML conversation logging. Forwards requests from clients to an upstream LLM service, records conversation history with per-message timestamps, and supports both streaming and non-streaming requests.

## Features

- **Transparent proxying** — `/v1/chat/completions`, `/v1/completions`, and catch-all forwarding for any other endpoint (e.g. `/v1/models`)
- **Streaming & non-streaming** — SSE stream reconstruction and passthrough
- **ChatML session logging** — Multi-turn conversation tracking with prefix matching, output as JSON with per-message ISO timestamps
- **Persistent config** — Parameters auto-saved to `llm_proxy.ini` via `configargparse`
- **Graceful shutdown** — Dumps pending ChatML sessions on SIGINT/SIGTERM

## Installation

```bash
pip install fastapi uvicorn configargparse httpx
```

## Quick Start

```bash
python llm_proxy.py --base-url http://localhost:8000 --api-key sk-your-key --log-chatml
```

This starts the proxy on `0.0.0.0:8030`, forwarding to `http://localhost:8000` with ChatML logging enabled.

Then point your OpenAI client at `http://localhost:8030/v1`.

## Configuration

All parameters can be set via command line or `llm_proxy.ini`. Command-line values override the config file and are written back on exit.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--host` | `0.0.0.0` | Proxy listen address |
| `--port` | `8030` | Proxy listen port |
| `--base-url` | *(required)* | Upstream LLM service URL (e.g. `http://localhost:8000`) |
| `--api-key` | `""` | Upstream API key — replaces any key sent by the client |
| `--log-folder` | `./logs/` | Directory for logs and ChatML output |
| `--log-chatml` | `false` | Enable ChatML conversation recording |
| `--session-name` | `sess_MMdd_HHmmss` | Name for the first ChatML session |

The config file is `llm_proxy.ini` in the working directory.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/proxyhealth` | Health check — returns `{"status": "ok"}` |
| POST | `/newsession` | Dump current ChatML sessions to file, reset session name. Body: `{"session_name": "..."}` |
| POST | `/v1/chat/completions` | Forward to upstream, record in ChatML session |
| POST | `/v1/completions` | Legacy completions endpoint |
| * | `/{path}` | Catch-all — forwards any other request to upstream |

## ChatML Output

When `--log-chatml` is enabled, conversation history is saved as JSON files in `--log-folder`:

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

- Messages use standard OpenAI roles (`system`, `user`, `assistant`, `tool`)
- Each message has an ISO 8601 `timestamp` — empty string for messages the proxy didn't directly witness
- `remarks.incomplete` is `true` when a conversation was cut off mid-turn
- Tool calls are stored in the `tool_calls` field of assistant messages
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
