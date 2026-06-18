import copy
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



def _decompress_body(data: bytes, content_encoding: str) -> bytes:
    """Decompress response body for ChatML reconstruction."""
    if not content_encoding or not data:
        return data
    ce = content_encoding.lower().strip()
    if ce in ("gzip", "x-gzip"):
        import gzip
        return gzip.decompress(data)
    if ce == "deflate":
        import zlib
        return zlib.decompress(data)
    if ce == "br":
        try:
            import brotli
            return brotli.decompress(data)
        except ImportError:
            pass
    return data


def _inject_temperature(body: dict, temperature_default: float) -> None:
    """Inject 'temperature' into *body* if default is non-negative and the
    body does not already contain a 'temperature' key."""
    if temperature_default >= 0 and "temperature" not in body:
        body["temperature"] = temperature_default


# ---------------------------------------------------------------------------
# SSE reconstruction helpers
# ---------------------------------------------------------------------------
def _reconstruct_chat_response(chunks: list[bytes]) -> dict | None:
    """Parse SSE chunks from a streaming /v1/chat/completions response
    and reconstruct a non-streaming response dict.  Returns None on parse
    failure."""
    # Join all raw bytes before decoding so that SSE data lines split
    # across aiter_bytes() chunk boundaries are not silently lost.
    full_text = b"".join(chunks).decode("utf-8", errors="replace")
    collected = {}
    for line in full_text.splitlines():
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
        if "prompt_token_ids" in obj:
            collected["prompt_token_ids"] = obj["prompt_token_ids"]
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
            if delta.get("reasoning_content"):
                c["message"].setdefault("reasoning_content", "")
                c["message"]["reasoning_content"] += delta["reasoning_content"]
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
            # --- accumulate RL fields from streaming chunks ---
            if "token_ids" in choice and choice["token_ids"] is not None:
                c.setdefault("token_ids", []).extend(choice["token_ids"])
            if "logprobs" in choice and choice["logprobs"]:
                c.setdefault("_logprobs_content", [])
                c["_logprobs_content"].extend(
                    choice["logprobs"].get("content", []))
    if "choices" not in collected:
        return None
    return collected


def _reconstruct_completion_response(chunks: list[bytes]) -> dict | None:
    """Parse SSE chunks from a streaming /v1/completions response."""
    # Join all raw bytes before decoding so that SSE data lines split
    # across aiter_bytes() chunk boundaries are not silently lost.
    full_text = b"".join(chunks).decode("utf-8", errors="replace")
    collected = {}
    for line in full_text.splitlines():
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
        if "prompt_token_ids" in obj:
            collected["prompt_token_ids"] = obj["prompt_token_ids"]
        for choice in obj.get("choices", []):
            idx = choice.get("index", 0)
            if "choices" not in collected:
                collected["choices"] = []
            while len(collected["choices"]) <= idx:
                collected["choices"].append({"text": "", "index": idx, "finish_reason": None})
            collected["choices"][idx]["text"] += choice.get("text", "")
            if choice.get("finish_reason"):
                collected["choices"][idx]["finish_reason"] = choice["finish_reason"]
            # --- accumulate RL fields from streaming chunks ---
            if "token_ids" in choice and choice["token_ids"] is not None:
                collected["choices"][idx].setdefault("token_ids", []).extend(
                    choice["token_ids"])
            if "logprobs" in choice and choice["logprobs"]:
                collected["choices"][idx].setdefault("_logprobs_content", [])
                collected["choices"][idx]["_logprobs_content"].extend(
                    choice["logprobs"].get("content", []))
    if "choices" not in collected:
        return None
    return collected


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app(base_url: str, api_key: str, session_manager: SessionManager,
               temperature_arg: float = -1.0,
               default_model: str | None = None,
               override_model: bool = False) -> FastAPI:
    app = FastAPI()
    upstream = base_url.rstrip("/")
    temperature_default = temperature_arg  # capture for handler closures
    # When --override-model is set and default_model is available, force all
    # requests to use the default model (equivalent to /change_override_model).
    model_override = [default_model] if (override_model and default_model) else [None]

    # --- /proxyhealth ---
    @app.get("/proxyhealth")
    async def proxyhealth():
        return {"status": "ok"}

    # --- /newsession ---
    @app.post("/newsession")
    async def newsession(request: Request):
        session_manager.dump_all()
        new_name = None
        new_path = None
        try:
            body = await request.json()
            new_name = body.get("session_name")
            new_path = body.get("session_path")
        except Exception:
            pass
        from datetime import datetime
        if not new_name:
            new_name = "sess_" + datetime.now().strftime("%m%d_%H%M%S")
        session_manager.session_name = new_name
        session_manager.session_path = new_path or session_manager.log_folder
        logger.info("newsession: switched to '%s' (path=%s)", new_name,
                    session_manager.session_path)
        return {"status": "ok", "session_name": new_name,
                "session_path": session_manager.session_path}

    # --- /session_chats ---
    @app.get("/session_chats")
    async def session_chats():
        if not session_manager.enabled:
            return JSONResponse(
                {"error": "chatml logging is disabled"}, status_code=400)
        chats = session_manager.get_session_chats()
        return JSONResponse(content=chats)

    # --- /change_override_model ---
    @app.post("/change_override_model")
    async def change_override_model(request: Request):
        try:
            body = await request.json()
        except Exception:
            body = {}
        model = body.get("model", "")
        if not model or (isinstance(model, str) and model.lower() == "none"):
            model_override[0] = None
            logger.info("change_override_model: cleared override (back to pass-through)")
            return {"status": "ok", "model_override": None}
        else:
            model_override[0] = model
            logger.info("change_override_model: override set to '%s'", model)
            return {"status": "ok", "model_override": model}

    # --- /v1/chat/completions ---
    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await _handle_chat_completions(request, upstream, api_key, session_manager,
                                              temperature_default, model_override,
                                              default_model)

    # --- /v1/completions ---
    @app.post("/v1/completions")
    async def completions(request: Request):
        return await _handle_completions(request, upstream, api_key, session_manager,
                                         temperature_default, model_override,
                                         default_model)

    # --- catch-all ---
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
    async def catchall(request: Request, path: str):
        return await _handle_catchall(request, upstream, api_key, path)

    return app


# ---------------------------------------------------------------------------
# Chat completions handler
# ---------------------------------------------------------------------------
async def _handle_chat_completions(request: Request, upstream: str, api_key: str,
                                  session_mgr: SessionManager,
                                  temperature_default: float = -1.0,
                                  model_override: list | None = None,
                                  default_model: str | None = None):
    body = await request.json()
    _inject_temperature(body, temperature_default)
    # --- model handling ---
    if model_override and model_override[0] is not None:
        body["model"] = model_override[0]
    elif body.get("model") in ("", "none", None) and default_model is not None:
        body["model"] = default_model
    messages = body.get("messages", [])
    tools = body.get("tools")
    is_stream = body.get("stream", False)
    req_ts = _now_iso()

    # --- session matching ---
    # Deep-copy messages before passing to session manager because
    # _normalize_msg_arguments mutates tool_calls[].function.arguments
    # from a JSON string to a dict in-place — which would corrupt the
    # request body sent to DeepSeek.
    session, match_len = session_mgr.find_matching_session(messages)
    if session is None:
        session = session_mgr.create_session(copy.deepcopy(messages), req_ts, tools)
    else:
        session_mgr.append_request_messages(session, copy.deepcopy(messages), match_len, req_ts, tools)

    # --- inject RL parameters ---
    if session_mgr.rl_enabled and session_mgr.enabled:
        body.setdefault("logprobs", True)
        body.setdefault("return_token_ids", True)

    # --- forward to upstream ---
    headers = _build_upstream_headers(request, api_key)
    body_stripped = _strip_openai_key(body)
    url = f"{upstream}/chat/completions"
    logger.debug("-> chat/completions stream=%s body=%s", is_stream,
                 json.dumps(body_stripped, ensure_ascii=False)[:500])

    try:
        if is_stream:
            return await _stream_forward(url, headers, body, session_mgr, session)
        else:
            return await _nonstream_forward(url, headers, body, session_mgr, session)
    except httpx.HTTPStatusError as e:
        # Read streaming response body so we can log the error details.
        try:
            await e.response.aread()
            upstream_text = e.response.text[:1000]
        except Exception:
            upstream_text = "<failed to read response body>"
        logger.error("upstream error %s for %s: %s\n  request body: %s",
                     e.response.status_code, url,
                     upstream_text,
                     json.dumps(body_stripped, ensure_ascii=False)[:2000])
        return Response(content=e.response.content, status_code=e.response.status_code,
                        headers=dict(e.response.headers))
    except Exception as e:
        logger.error("proxy error: %s", e)
        return JSONResponse({"error": str(e)}, status_code=502)


# ---------------------------------------------------------------------------
# Completions handler (legacy)
# ---------------------------------------------------------------------------
async def _handle_completions(request: Request, upstream: str, api_key: str,
                              session_mgr: SessionManager,
                              temperature_default: float = -1.0,
                              model_override: list | None = None,
                              default_model: str | None = None):
    body = await request.json()
    _inject_temperature(body, temperature_default)
    # --- model handling ---
    if model_override and model_override[0] is not None:
        body["model"] = model_override[0]
    elif body.get("model") in ("", "none", None) and default_model is not None:
        body["model"] = default_model

    # --- inject RL parameters ---
    if session_mgr.rl_enabled and session_mgr.enabled:
        body.setdefault("logprobs", True)
        body.setdefault("return_token_ids", True)

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
    url = f"{upstream}/completions"
    logger.debug("-> completions stream=%s prompt=%.200s", is_stream, prompt)

    try:
        if is_stream:
            return await _stream_forward_completions(url, headers, body, session_mgr, session)
        else:
            return await _nonstream_forward_completions(url, headers, body, session_mgr, session)
    except httpx.HTTPStatusError as e:
        # Read streaming response body so we can log the error details.
        try:
            await e.response.aread()
            upstream_text = e.response.text[:1000]
        except Exception:
            upstream_text = "<failed to read response body>"
        logger.error("upstream error %s for %s: %s\n  request body: %s",
                     e.response.status_code, url,
                     upstream_text,
                     json.dumps(body, ensure_ascii=False)[:2000])
        return Response(content=e.response.content, status_code=e.response.status_code,
                        headers=dict(e.response.headers))
    except Exception as e:
        logger.error("proxy error: %s", e)
        return JSONResponse({"error": str(e)}, status_code=502)


def _find_completion_session(session_mgr: SessionManager, prompt: str):
    """Prefix match for legacy completions — compares accumulated prompt text."""
    if not session_mgr.enabled:
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

    # Use client.send() to capture response headers before streaming,
    # so we can forward content-encoding to the client.
    client = httpx.AsyncClient(timeout=300)
    resp = await client.send(
        client.build_request("POST", url, json=body, headers=headers),
        stream=True,
    )
    try:
        resp.raise_for_status()
    except Exception:
        await client.aclose()
        raise

    resp_ce = resp.headers.get("content-encoding", "")

    async def generator():
        try:
            async for chunk in resp.aiter_bytes():
                chunks.append(chunk)
                yield chunk
        finally:
            await resp.aclose()
            await client.aclose()

    async def wrapper():
        async for chunk in generator():
            yield chunk
        resp_ts = _now_iso()
        raw = b"".join(chunks)
        decompressed = _decompress_body(raw, resp_ce)
        reconstructed = _reconstruct_chat_response([decompressed])
        if reconstructed:
            _record_chat_response(session_mgr, session, reconstructed, resp_ts)
        logger.info("chat stream %d bytes (%d chunks)", sum(len(c) for c in chunks), len(chunks))

    resp_headers = {}
    if resp_ce:
        resp_headers["content-encoding"] = resp_ce
    return StreamingResponse(wrapper(), media_type="text/event-stream",
                             headers=resp_headers)


async def _stream_forward_completions(url: str, headers: dict,
                                      body: dict, session_mgr: SessionManager, session: dict):
    chunks: list[bytes] = []

    client = httpx.AsyncClient(timeout=300)
    resp = await client.send(
        client.build_request("POST", url, json=body, headers=headers),
        stream=True,
    )
    try:
        resp.raise_for_status()
    except Exception:
        await client.aclose()
        raise

    resp_ce = resp.headers.get("content-encoding", "")

    async def generator():
        try:
            async for chunk in resp.aiter_bytes():
                chunks.append(chunk)
                yield chunk
        finally:
            await resp.aclose()
            await client.aclose()

    async def wrapper():
        async for chunk in generator():
            yield chunk
        resp_ts = _now_iso()
        raw = b"".join(chunks)
        decompressed = _decompress_body(raw, resp_ce)
        reconstructed = _reconstruct_completion_response([decompressed])
        if reconstructed:
            _record_completion_response(session_mgr, session, reconstructed, resp_ts)
        logger.info("completions stream %d bytes (%d chunks)", sum(len(c) for c in chunks), len(chunks))

    resp_headers = {}
    if resp_ce:
        resp_headers["content-encoding"] = resp_ce
    return StreamingResponse(wrapper(), media_type="text/event-stream",
                             headers=resp_headers)


# ---------------------------------------------------------------------------
# Response recording
# ---------------------------------------------------------------------------
def _record_chat_response(session_mgr: SessionManager, session: dict,
                          resp_body: dict, timestamp: str):
    if not session_mgr.enabled:
        return
    prompt_ids = resp_body.get("prompt_token_ids")
    for choice in resp_body.get("choices", []):
        msg = choice.get("message", {})
        if msg:
            msg = dict(msg)
            if session_mgr.rl_enabled:
                if prompt_ids is not None:
                    msg["prompt_ids"] = prompt_ids
                # Completion token ids — from non-streaming or reconstructed streaming
                token_ids = choice.get("token_ids")
                if token_ids is not None:
                    msg["completion_ids"] = token_ids
                # Logprobs — from non-streaming response
                logprobs = choice.get("logprobs")
                if logprobs and logprobs.get("content"):
                    msg["logprobs"] = [item["logprob"] for item in logprobs["content"]]
                # Logprobs — from reconstructed streaming (accumulated via _logprobs_content)
                lp_content = choice.get("_logprobs_content")
                if lp_content:
                    msg["logprobs"] = [item["logprob"] for item in lp_content]
            session_mgr.append_response(session, msg, timestamp)


def _record_completion_response(session_mgr: SessionManager, session: dict,
                                resp_body: dict, timestamp: str):
    if not session_mgr.enabled:
        return
    prompt_ids = resp_body.get("prompt_token_ids")
    for choice in resp_body.get("choices", []):
        text = choice.get("text", "")
        if text:
            msg = {"role": "assistant", "content": text}
            if session_mgr.rl_enabled:
                if prompt_ids is not None:
                    msg["prompt_ids"] = prompt_ids
                token_ids = choice.get("token_ids")
                if token_ids is not None:
                    msg["completion_ids"] = token_ids
                logprobs = choice.get("logprobs")
                if logprobs and logprobs.get("content"):
                    msg["logprobs"] = [item["logprob"] for item in logprobs["content"]]
                lp_content = choice.get("_logprobs_content")
                if lp_content:
                    msg["logprobs"] = [item["logprob"] for item in lp_content]
            session_mgr.append_response(session, msg, timestamp)


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
