#!/usr/bin/env python3
"""Integration test for llm_proxy.

Starts a mock OpenAI-compatible upstream in a thread, runs the proxy, and
verifies all endpoints.  Can also test against a real upstream by passing
--real-upstream URL.
"""

import argparse
import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


# ---------------------------------------------------------------------------
# Mock upstream server
# ---------------------------------------------------------------------------
def make_mock_upstream():
    app = FastAPI()

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": "mock-model", "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages = body.get("messages", [])
        last_msg = messages[-1].get("content", "") if messages else ""
        is_stream = body.get("stream", False)
        has_tools = bool(body.get("tools"))

        if is_stream:
            return StreamingResponse(
                _stream_chat_response(last_msg, has_tools, body.get("model", "mock")),
                media_type="text/event-stream",
            )
        else:
            return JSONResponse(
                _nonstream_chat_response(last_msg, has_tools, body.get("model", "mock"))
            )

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        prompt = body.get("prompt", "")
        is_stream = body.get("stream", False)

        if is_stream:
            return StreamingResponse(
                _stream_completion_response(prompt, body.get("model", "mock")),
                media_type="text/event-stream",
            )
        else:
            return JSONResponse(
                _nonstream_completion_response(prompt, body.get("model", "mock"))
            )

    @app.get("/v1/error_test")
    async def error_test():
        return JSONResponse(
            {"error": {"message": "test error", "type": "test"}}, status_code=500
        )

    return app


def _nonstream_chat_response(user_text, has_tools, model):
    if has_tools:
        return {
            "id": "chatcmpl-mock-001",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_mock_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Beijing"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
        }
    return {
        "id": "chatcmpl-mock-001",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"[mock reply to: {user_text}]",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


async def _stream_chat_response(user_text, has_tools, model):
    rid = "chatcmpl-mock-stream"
    yield f'data: {json.dumps({"id": rid, "object": "chat.completion.chunk", "model": model, "choices": [{"index": 0, "delta": {"role": "assistant"}}]})}\n\n'
    if has_tools:
        yield f'data: {{"id":"{rid}","object":"chat.completion.chunk","model":"{model}","choices":[{{"index":0,"delta":{{"tool_calls":[{{"index":0,"id":"call_s1","type":"function","function":{{"name":"get_weather"}}}}]}}}}]}}\n\n'
        yield f'data: {{"id":"{rid}","object":"chat.completion.chunk","model":"{model}","choices":[{{"index":0,"delta":{{"tool_calls":[{{"index":0,"function":{{"arguments":"{{\\"city\\":\\"Beijing\\"}}"}}}}]}}}}]}}\n\n'
    else:
        for w in ["Hello", " there", " from", " mock!"]:
            yield f'data: {json.dumps({"id": rid, "object": "chat.completion.chunk", "model": model, "choices": [{"index": 0, "delta": {"content": w}}]})}\n\n'
    yield f'data: {json.dumps({"id": rid, "object": "chat.completion.chunk", "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})}\n\n'
    yield "data: [DONE]\n\n"


def _nonstream_completion_response(prompt, model):
    return {
        "id": "cmpl-mock-001",
        "object": "text_completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": f" completion for: {prompt[-30:]}",
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


async def _stream_completion_response(prompt, model):
    rid = "cmpl-mock-stream"
    for word in [" comp", "letion", " mock"]:
        yield f'data: {json.dumps({"id": rid, "object": "text_completion.chunk", "model": model, "choices": [{"index": 0, "text": word}]})}\n\n'
    yield f'data: {json.dumps({"id": rid, "object": "text_completion.chunk", "model": model, "choices": [{"index": 0, "text": "", "finish_reason": "stop"}]})}\n\n'
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_port(host, port, timeout=10):
    """Poll until the server accepts connections."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            time.sleep(0.1)
    return False


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
class TestProxy:
    def __init__(self, proxy_url, chatml_dir):
        self.proxy_url = proxy_url.rstrip("/")
        self.chatml_dir = chatml_dir
        self.client = httpx.Client(timeout=30)

    def _post(self, path, json_body=None):
        return self.client.post(f"{self.proxy_url}{path}", json=json_body or {})

    def _get(self, path):
        return self.client.get(f"{self.proxy_url}{path}")

    def _stream_post(self, path, json_body):
        parts = []
        with self.client.stream("POST", f"{self.proxy_url}{path}", json=json_body) as resp:
            for chunk in resp.iter_bytes():
                parts.append(chunk.decode("utf-8", errors="replace"))
        return "".join(parts)

    # ---- tests ----
    def test_proxyhealth(self):
        r = self._get("/proxyhealth")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        print("  PASS proxyhealth")

    def test_newsession(self):
        r = self._post("/newsession", {"session_name": "test_sess"})
        assert r.status_code == 200
        assert r.json()["session_name"] == "test_sess"
        r2 = self._post("/newsession")
        assert r2.status_code == 200
        assert r2.json()["session_name"].startswith("sess_")
        print("  PASS newsession")

    def test_nonstream_chat(self):
        body = {
            "model": "mock-model",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
            "stream": False,
        }
        r = self._post("/v1/chat/completions", body)
        assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text[:200]}"
        data = r.json()
        assert "choices" in data
        assert data["choices"][0]["message"]["role"] == "assistant"
        print("  PASS nonstream chat")

    def test_stream_chat(self):
        body = {
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }
        text = self._stream_post("/v1/chat/completions", body)
        assert "data: " in text
        assert "[DONE]" in text
        print("  PASS stream chat")

    def test_multi_round(self):
        body1 = {
            "model": "mock",
            "messages": [{"role": "user", "content": "Q1"}],
            "stream": False,
        }
        r1 = self._post("/v1/chat/completions", body1)
        assert r1.status_code == 200
        a1 = r1.json()["choices"][0]["message"]["content"]

        body2 = {
            "model": "mock",
            "messages": [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": a1},
                {"role": "user", "content": "Q2"},
            ],
            "stream": False,
        }
        r2 = self._post("/v1/chat/completions", body2)
        assert r2.status_code == 200
        a2 = r2.json()["choices"][0]["message"]["content"]

        body3 = {
            "model": "mock",
            "messages": [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": a1},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": a2},
                {"role": "user", "content": "Q3"},
            ],
            "stream": False,
        }
        r3 = self._post("/v1/chat/completions", body3)
        assert r3.status_code == 200
        print("  PASS multi-round")

    def test_tool_calls(self):
        body = {
            "model": "mock",
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
            "stream": False,
        }
        r = self._post("/v1/chat/completions", body)
        assert r.status_code == 200
        msg = r.json()["choices"][0]["message"]
        assert msg["tool_calls"] is not None
        assert len(msg["tool_calls"]) > 0
        print("  PASS tool calls non-stream")

    def test_tool_calls_stream(self):
        body = {
            "model": "mock",
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
            "stream": True,
        }
        text = self._stream_post("/v1/chat/completions", body)
        assert "get_weather" in text
        print("  PASS tool calls stream")

    def test_catchall(self):
        r = self._get("/v1/models")
        assert r.status_code == 200
        assert "data" in r.json()
        print("  PASS catchall /v1/models")

    def test_upstream_error(self):
        r = self._get("/v1/error_test")
        assert r.status_code == 500
        print("  PASS upstream error passthrough")

    def test_newsession_dump(self):
        import glob
        # Clear any existing json files
        for f in glob.glob(os.path.join(self.chatml_dir, "*.json")):
            os.remove(f)

        body = {
            "model": "mock",
            "messages": [{"role": "user", "content": "Dump test"}],
            "stream": False,
        }
        self._post("/v1/chat/completions", body)
        self._post("/newsession", {"session_name": "test_dump_sess"})

        # dump_all() runs before the name change, so file uses the OLD session name
        files = glob.glob(os.path.join(self.chatml_dir, "*.json"))
        assert len(files) >= 1, f"No ChatML files found in {self.chatml_dir}"
        filepath = files[0]
        with open(filepath) as f:
            data = json.load(f)
        assert "messages" in data
        assert "remarks" in data
        assert len(data["messages"]) >= 2
        print("  PASS newsession dump")


def run_tests(tester):
    tests = [
        ("proxyhealth", tester.test_proxyhealth),
        ("newsession", tester.test_newsession),
        ("nonstream_chat", tester.test_nonstream_chat),
        ("stream_chat", tester.test_stream_chat),
        ("multi_round", tester.test_multi_round),
        ("tool_calls", tester.test_tool_calls),
        ("tool_calls_stream", tester.test_tool_calls_stream),
        ("catchall", tester.test_catchall),
        ("upstream_error", tester.test_upstream_error),
        ("newsession_dump", tester.test_newsession_dump),
    ]
    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
    return failed == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Test llm_proxy")
    parser.add_argument("--proxy-url", default=None, help="URL of already-running proxy")
    parser.add_argument("--real-upstream", default=None, help="Use a real upstream instead of mock")
    args = parser.parse_args()

    if args.proxy_url:
        tester = TestProxy(args.proxy_url, "./logs")
        success = run_tests(tester)
        sys.exit(0 if success else 1)

    cwd = os.path.dirname(os.path.abspath(__file__))

    # --- Start mock upstream ---
    if args.real_upstream:
        upstream_url = args.real_upstream
    else:
        mock_port = find_free_port()
        upstream_url = f"http://127.0.0.1:{mock_port}"

        # Kill anything on the port (from a previous run)
        subprocess.run(["fuser", "-k", f"{mock_port}/tcp"], stderr=subprocess.DEVNULL)

        print(f"Starting mock upstream on port {mock_port} ...")
        mock_proc = subprocess.Popen(
            [sys.executable, "-u", "-c",
             f"import sys; sys.path.insert(0, {cwd!r}); "
             f"from test_proxy import make_mock_upstream; "
             f"import uvicorn; "
             f"uvicorn.run(make_mock_upstream(), host='127.0.0.1', port={mock_port}, log_level='warning')"],
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        if not wait_for_port("127.0.0.1", mock_port, timeout=5):
            stderr = mock_proc.stderr.read().decode() if mock_proc.stderr else ""
            print(f"Mock upstream failed to start:\n{stderr}")
            mock_proc.kill()
            mock_proc.wait()
            sys.exit(1)

    # --- Start proxy ---
    proxy_port = find_free_port()
    proxy_url = f"http://127.0.0.1:{proxy_port}"
    log_dir = tempfile.mkdtemp(prefix="llm_proxy_test_")

    subprocess.run(["fuser", "-k", f"{proxy_port}/tcp"], stderr=subprocess.DEVNULL)

    print(f"Starting llm_proxy on port {proxy_port} -> upstream {upstream_url}")
    proxy_proc = subprocess.Popen(
        [sys.executable, "-u", "llm_proxy.py",
         "--host", "127.0.0.1",
         "--port", str(proxy_port),
         "--base-url", upstream_url,
         "--log-folder", log_dir,
         "--log-chatml", "multi",
         "--session-name", "test_sess"],
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    if not wait_for_port("127.0.0.1", proxy_port, timeout=5):
        stderr = proxy_proc.stderr.read().decode() if proxy_proc.stderr else ""
        print(f"Proxy failed to start:\n{stderr}")
        proxy_proc.kill()
        proxy_proc.wait()
        if not args.real_upstream:
            mock_proc.kill()
            mock_proc.wait()
        sys.exit(1)

    print(f"Log dir: {log_dir}")

    try:
        tester = TestProxy(proxy_url, log_dir)
        success = run_tests(tester)
    finally:
        for p in [proxy_proc] + ([mock_proc] if not args.real_upstream else []):
            p.send_signal(signal.SIGTERM)
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        print(f"Cleaned up. Logs at: {log_dir}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
