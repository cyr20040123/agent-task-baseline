import json
import os
from datetime import datetime, timezone


def _normalize_tool_calls(tool_calls):
    """Normalize tool_calls for stable comparison — parse arguments JSON."""
    if not tool_calls:
        return None
    result = []
    for tc in tool_calls:
        func = tc.get("function", {})
        args = func.get("arguments", "{}")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        result.append({
            "id": tc.get("id", ""),
            "type": tc.get("type", "function"),
            "function": {"name": func.get("name", ""), "arguments": args},
        })
    return result


# Fields used for matching — only standard OpenAI message fields.
# Model-specific extras (reasoning, refusal, annotations, audio, etc.)
# are excluded so that the client's echoed messages match responses.
_MATCH_FIELDS = {"role", "content", "tool_calls", "tool_call_id", "name"}


def _canonical(msg):
    """Serialize a message dict to a canonical JSON string for comparison.
    Strips 'timestamp' and non-standard fields, normalizes tool_calls.
    Null values and empty tool_calls are omitted."""
    d = {}
    for k, v in msg.items():
        if k == "timestamp":
            continue
        if k not in _MATCH_FIELDS:
            continue
        if k == "tool_calls":
            n = _normalize_tool_calls(v)
            if n:
                d[k] = n
        elif v is not None:
            d[k] = v
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


class SessionManager:
    def __init__(self, log_folder, session_name, log_chatml_enabled):
        self.log_folder = log_folder
        self.session_name = session_name
        self.log_chatml_enabled = log_chatml_enabled
        self.sessions = []

    # ------------------------------------------------------------------
    # Prefix matching
    # ------------------------------------------------------------------
    def find_matching_session(self, request_messages):
        """Return (session, match_len) where session.messages[:match_len]
        is a prefix of request_messages.  Comparison ignores timestamps
        and normalises tool-call arguments."""
        if not self.log_chatml_enabled:
            return None, 0
        for sess in self.sessions:
            sess_msgs = sess["messages"]
            if len(sess_msgs) > len(request_messages):
                continue
            ok = True
            for i, sm in enumerate(sess_msgs):
                if _canonical(sm) != _canonical(request_messages[i]):
                    ok = False
                    break
            if ok:
                return sess, len(sess_msgs)
        return None, 0

    # ------------------------------------------------------------------
    # Session creation / update
    # ------------------------------------------------------------------
    def create_session(self, request_messages, timestamp, tools=None):
        """Create a new session.  Only the *last* message is considered
        directly observed; earlier messages get an empty timestamp because
        the proxy did not witness their arrival."""
        session = {"messages": [], "tools": []}
        if tools:
            session["tools"] = list(tools)
        for i, msg in enumerate(request_messages):
            ts = timestamp if i == len(request_messages) - 1 else ""
            session["messages"].append({**msg, "timestamp": ts})
        self.sessions.append(session)
        return session

    def append_request_messages(self, session, request_messages, match_len, timestamp,
                                tools=None):
        """Append suffix messages (those beyond match_len) to the session.
        Each new message gets the request-arrival timestamp.
        New tool definitions are merged in (deduplicated by function name)."""
        new_msgs = request_messages[match_len:]
        for msg in new_msgs:
            session["messages"].append({**msg, "timestamp": timestamp})
        if tools:
            known = {t.get("function", {}).get("name") for t in session["tools"]}
            for t in tools:
                name = t.get("function", {}).get("name")
                if name and name not in known:
                    session["tools"].append(t)
                    known.add(name)

    def append_response(self, session, response_message, timestamp):
        """Append an assistant response message with its own timestamp."""
        session["messages"].append({**response_message, "timestamp": timestamp})

    # ------------------------------------------------------------------
    # Dump to ChatML JSON
    # ------------------------------------------------------------------
    def dump_all(self):
        if not self.log_chatml_enabled:
            self.sessions = []
            return
        os.makedirs(self.log_folder, exist_ok=True)
        for i, sess in enumerate(self.sessions):
            suffix = f"_{i}" if len(self.sessions) > 1 else ""
            self._dump_session(sess, suffix)
        self.sessions = []

    def _dump_session(self, sess, suffix=""):
        msgs = sess["messages"]
        chatml_msgs, incomplete = self._build_chatml(msgs)
        if chatml_msgs is None:
            return

        output = {
            "messages": chatml_msgs,
            "remarks": {"incomplete": incomplete},
        }
        if sess.get("tools"):
            output["tools"] = sess["tools"]

        filepath = os.path.join(self.log_folder, f"{self.session_name}{suffix}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    def _build_chatml(self, messages):
        """If the session ends with an assistant message it is complete.
        Otherwise truncate to the longest prefix ending with an assistant
        message and mark incomplete."""
        if not messages:
            return None, False

        if messages[-1].get("role") == "assistant":
            return messages, False

        # Truncate to last assistant
        trunc = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                trunc = i
                break
        if trunc == -1:
            return None, False  # no assistant message at all — skip

        return messages[: trunc + 1], True
