import json
import logging

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse, Response

from chatml_session import SessionManager, _now_iso

logger = logging.getLogger("llm_proxy")


def _build_upstream_headers(request: Request, api_key: str) -> dict:
    """Copy select headers from the incoming request, replace Authorization."""
    headers = {}
    for key in ("content-type", "accept", "accept-encoding"):
        if key in request.headers:
            headers[key] = request.headers[key]
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    elif "authorization" in request.headers:
        headers["authorization"] = request.headers["authorization"]
    return headers


def _strip_openai_key(body: dict) -> dict:
    """Shallow-copy and redact api_key for logging."""
    d = {k: v for k, v in body.items() if k != "api_key"}
    return d


# ---------------------------------------------------------------------------
# SSE reconstruction helpers
# ---------------------------------------------------------------------------
def _reconstruct_chat_response(chunks: list[bytes]) -> dict | None:
    """Parse SSE chunks from a streaming /v1/chat/completions response
    and reconstruct a non-streaming response dict.  Returns None on parse
    failure."""
    collected = {}
    for chunk in chunks:
        text = chunk.decode("utf-8", errors="replace")
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if "id" in obj:
                collected.setdefault("id", obj["id"])
            if "object" in obj:
                collected.setdefault("object", obj["object"].replace(".chunk", ""))
            if "model" in obj and "model" not in collected:
                collected["model"] = obj["model"]
            if "usage" in obj and obj["usage"]:
                collected["usage"] = obj["usage"]
            for choice in obj.get("choices", []):
                idx = choice.get("index", 0)
                if "choices" not in collected:
                    collected["choices"] = []
                while len(collected["choices"]) <= idx:
                    collected["choices"].append({
                        "index": idx,
                        "message": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    })
                c = collected["choices"][idx]
                delta = choice.get("delta", {})
                if delta.get("role"):
                    c["message"]["role"] = delta["role"]
                if delta.get("content"):
                    c["message"]["content"] += delta["content"]
                if delta.get("reasoning"):
                    c["message"].setdefault("reasoning", "")
                    c["message"]["reasoning"] += delta["reasoning"]
                if delta.get("tool_calls"):
                    tc_map = c["message"].setdefault("tool_calls", [])
                    for tc in delta["tool_calls"]:
                        tci = tc.get("index", 0)
                        while len(tc_map) <= tci:
                            tc_map.append({
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            })
                        if tc.get("id"):
                            tc_map[tci]["id"] = tc["id"]
                        if tc.get("function", {}).get("name"):
                            tc_map[tci]["function"]["name"] = tc["function"]["name"]
                        if tc.get("function", {}).get("arguments"):
                            tc_map[tci]["function"]["arguments"] += tc["function"]["arguments"]
                if choice.get("finish_reason"):
                    c["finish_reason"] = choice["finish_reason"]
    if "choices" not in collected:
        return None
    # Clean up: content → null when only tool_calls are present (no text, no reasoning)
    for c in collected.get("choices", []):
        msg = c.get("message", {})
        if msg.get("tool_calls") and not msg.get("content") and not msg.get("reasoning"):
            msg["content"] = None
    return collected


def _reconstruct_completion_response(chunks: list[bytes]) -> dict | None:
    """Parse SSE chunks from a streaming /v1/completions response."""
    collected = {}
    for chunk in chunks:
        text = chunk.decode("utf-8", errors="replace")
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if "id" in obj:
                collected.setdefault("id", obj["id"])
            if "object" in obj:
                collected.setdefault("object", obj["object"].replace(".chunk", ""))
            if "model" in obj and "model" not in collected:
                collected["model"] = obj["model"]
            if "usage" in obj and obj["usage"]:
                collected["usage"] = obj["usage"]
            for choice in obj.get("choices", []):
                idx = choice.get("index", 0)
                if "choices" not in collected:
                    collected["choices"] = []
                while len(collected["choices"]) <= idx:
                    collected["choices"].append({"text": "", "index": idx, "finish_reason": None})
                collected["choices"][idx]["text"] += choice.get("text", "")
                if choice.get("finish_reason"):
                    collected["choices"][idx]["finish_reason"] = choice["finish_reason"]
    if "choices" not in collected:
        return None
    return collected


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app(base_url: str, api_key: str, session_manager: SessionManager) -> FastAPI:
    app = FastAPI()
    upstream = base_url.rstrip("/")

    # --- /proxyhealth ---
    @app.get("/proxyhealth")
    async def proxyhealth():
        return {"status": "ok"}

    # --- /newsession ---
    @app.post("/newsession")
    async def newsession(request: Request):
        session_manager.dump_all()
        new_name = None
        try:
            body = await request.json()
            new_name = body.get("session_name")
        except Exception:
            pass
        from datetime import datetime
        if not new_name:
            new_name = "sess_" + datetime.now().strftime("%m%d_%H%M%S")
        session_manager.session_name = new_name
        logger.info("newsession: switched to '%s'", new_name)
        return {"status": "ok", "session_name": new_name}

    # --- /v1/chat/completions ---
    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await _handle_chat_completions(request, upstream, api_key, session_manager)

    # --- /v1/completions ---
    @app.post("/v1/completions")
    async def completions(request: Request):
        return await _handle_completions(request, upstream, api_key, session_manager)

    # --- catch-all ---
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
    async def catchall(request: Request, path: str):
        return await _handle_catchall(request, upstream, api_key, path)

    return app


# ---------------------------------------------------------------------------
# Chat completions handler
# ---------------------------------------------------------------------------
async def _handle_chat_completions(request: Request, upstream: str, api_key: str,
                                  session_mgr: SessionManager):
    body = await request.json()
    messages = body.get("messages", [])
    tools = body.get("tools")
    is_stream = body.get("stream", False)
    req_ts = _now_iso()

    # --- session matching ---
    session, match_len = session_mgr.find_matching_session(messages)
    if session is None:
        session = session_mgr.create_session(messages, req_ts, tools)
    else:
        session_mgr.append_request_messages(session, messages, match_len, req_ts, tools)

    # --- forward to upstream ---
    headers = _build_upstream_headers(request, api_key)
    body_stripped = _strip_openai_key(body)
    url = f"{upstream}/v1/chat/completions"
    logger.debug("-> chat/completions stream=%s body=%s", is_stream,
                 json.dumps(body_stripped, ensure_ascii=False)[:500])

    try:
        if is_stream:
            return await _stream_forward(url, headers, body, session_mgr, session)
        else:
            return await _nonstream_forward(url, headers, body, session_mgr, session)
    except httpx.HTTPStatusError as e:
        logger.error("upstream error %s: %s", e.response.status_code, e.response.text[:500])
        return Response(content=e.response.content, status_code=e.response.status_code,
                        headers=dict(e.response.headers))
    except Exception as e:
        logger.error("proxy error: %s", e)
        return JSONResponse({"error": str(e)}, status_code=502)


# ---------------------------------------------------------------------------
# Completions handler (legacy)
# ---------------------------------------------------------------------------
async def _handle_completions(request: Request, upstream: str, api_key: str,
                              session_mgr: SessionManager):
    body = await request.json()
    prompt = body.get("prompt", "")
    is_stream = body.get("stream", False)
    req_ts = _now_iso()

    # Simple string-prefix matching for completions
    session, match_len = _find_completion_session(session_mgr, prompt)
    if session is None:
        session = session_mgr.create_session(
            [{"role": "user", "content": prompt}], req_ts
        )
    else:
        new_text = prompt[match_len:]
        if new_text:
            session_mgr.append_request_messages(
                session, [{"role": "user", "content": new_text}], 0, req_ts
            )

    headers = _build_upstream_headers(request, api_key)
    url = f"{upstream}/v1/completions"
    logger.debug("-> completions stream=%s prompt=%.200s", is_stream, prompt)

    try:
        if is_stream:
            return await _stream_forward_completions(url, headers, body, session_mgr, session)
        else:
            return await _nonstream_forward_completions(url, headers, body, session_mgr, session)
    except httpx.HTTPStatusError as e:
        logger.error("upstream error %s: %s", e.response.status_code, e.response.text[:500])
        return Response(content=e.response.content, status_code=e.response.status_code,
                        headers=dict(e.response.headers))
    except Exception as e:
        logger.error("proxy error: %s", e)
        return JSONResponse({"error": str(e)}, status_code=502)


def _find_completion_session(session_mgr: SessionManager, prompt: str):
    """Prefix match for legacy completions — compares accumulated prompt text."""
    if not session_mgr.log_chatml_enabled:
        return None, 0
    for sess in session_mgr.sessions:
        accum = ""
        for msg in sess["messages"]:
            if msg.get("role") == "user":
                accum += msg["content"]
        if prompt.startswith(accum):
            return sess, len(accum)
    return None, 0


# ---------------------------------------------------------------------------
# Non-streaming helpers
# ---------------------------------------------------------------------------
async def _nonstream_forward(url: str, headers: dict,
                             body: dict, session_mgr: SessionManager, session: dict):
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        resp_ts = _now_iso()
        resp_body = resp.json()
    _record_chat_response(session_mgr, session, resp_body, resp_ts)
    logger.info("chat non-stream %d bytes", len(resp.content))
    return JSONResponse(content=resp_body, status_code=resp.status_code)


async def _nonstream_forward_completions(url: str, headers: dict,
                                         body: dict, session_mgr: SessionManager, session: dict):
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        resp_ts = _now_iso()
        resp_body = resp.json()
    _record_completion_response(session_mgr, session, resp_body, resp_ts)
    logger.info("completions non-stream %d bytes", len(resp.content))
    return JSONResponse(content=resp_body, status_code=resp.status_code)


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------
async def _stream_forward(url: str, headers: dict,
                          body: dict, session_mgr: SessionManager, session: dict):
    chunks: list[bytes] = []

    async def generator():
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():
                    chunks.append(chunk)
                    yield chunk

    async def wrapper():
        async for chunk in generator():
            yield chunk
        resp_ts = _now_iso()
        reconstructed = _reconstruct_chat_response(chunks)
        if reconstructed:
            _record_chat_response(session_mgr, session, reconstructed, resp_ts)
        logger.info("chat stream %d bytes (%d chunks)", sum(len(c) for c in chunks), len(chunks))

    return StreamingResponse(wrapper(), media_type="text/event-stream")


async def _stream_forward_completions(url: str, headers: dict,
                                      body: dict, session_mgr: SessionManager, session: dict):
    chunks: list[bytes] = []

    async def generator():
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():
                    chunks.append(chunk)
                    yield chunk

    async def wrapper():
        async for chunk in generator():
            yield chunk
        resp_ts = _now_iso()
        reconstructed = _reconstruct_completion_response(chunks)
        if reconstructed:
            _record_completion_response(session_mgr, session, reconstructed, resp_ts)
        logger.info("completions stream %d bytes (%d chunks)", sum(len(c) for c in chunks), len(chunks))

    return StreamingResponse(wrapper(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Response recording
# ---------------------------------------------------------------------------
def _record_chat_response(session_mgr: SessionManager, session: dict,
                          resp_body: dict, timestamp: str):
    if not session_mgr.log_chatml_enabled:
        return
    for choice in resp_body.get("choices", []):
        msg = choice.get("message", {})
        if msg:
            session_mgr.append_response(session, dict(msg), timestamp)


def _record_completion_response(session_mgr: SessionManager, session: dict,
                                resp_body: dict, timestamp: str):
    if not session_mgr.log_chatml_enabled:
        return
    for choice in resp_body.get("choices", []):
        text = choice.get("text", "")
        if text:
            session_mgr.append_response(session, {"role": "assistant", "content": text}, timestamp)


# ---------------------------------------------------------------------------
# Catch-all
# ---------------------------------------------------------------------------
async def _handle_catchall(request: Request, upstream: str, api_key: str, path: str):
    headers = _build_upstream_headers(request, api_key)
    url = f"{upstream}/{path}"
    if request.url.query:
        url += "?" + request.url.query

    body = None
    if request.method in ("POST", "PUT", "PATCH"):
        body = await request.body()

    logger.info("catchall %s %s", request.method, url)
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.request(
                method=request.method, url=url, content=body, headers=headers
            )
            resp.raise_for_status()
        return Response(content=resp.content, status_code=resp.status_code,
                        headers=dict(resp.headers))
    except httpx.HTTPStatusError as e:
        logger.error("catchall upstream error %s: %s", e.response.status_code, e.response.text[:500])
        return Response(content=e.response.content, status_code=e.response.status_code,
                        headers=dict(e.response.headers))
    except Exception as e:
        logger.error("catchall proxy error: %s", e)
        return JSONResponse({"error": str(e)}, status_code=502)
