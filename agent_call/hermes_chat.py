#!/usr/bin/env python3
"""
CLI 或函数调用：运行 `hermes chat`，从重定向输出解析并写入 {chat_id}_hermes_f_output.txt（默认目录 ./hermes_output_logs/）。
输出四段：model、query、last_response、session_summary。
chat_id 格式：MMdd-HHmmss（strftime %m%d-%H%M%S）。
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# 与 CLI --output-dir 默认一致：相对当前工作目录
DEFAULT_OUTPUT_DIR = Path("hermes_output_logs")

HERMES_BOX_TOP = "╭─ ⚕ Hermes ─"
HERMES_BOX_BOTTOM = "╰────────"
RESUME_PREFIX = "Resume this session with:"
NOUS_RESEARCH = "· Nous Research"
QUERY_MARK = "\nQuery: "
QUERY_MARK_ALT = "Query: "
INIT_MARK = "\nInitializing"


def extract_model(text: str) -> str:
    """第一个「· Nous Research」左侧最近的「│」之后到该标记之前（均不含），再 strip。"""
    i = text.find(NOUS_RESEARCH)
    if i == -1:
        return ""
    j = text.rfind("│", 0, i)
    if j == -1:
        return ""
    return text[j + 1 : i].strip()


def extract_query(text: str) -> str:
    """第一个「\\nQuery: 」之后到「\\nInitializing」之前（均不含边界标记本体）。"""
    i = text.find(QUERY_MARK)
    start: int
    if i == -1:
        i = text.find(QUERY_MARK_ALT)
        if i == -1:
            return ""
        start = i + len(QUERY_MARK_ALT)
    else:
        start = i + len(QUERY_MARK)
    j = text.find(INIT_MARK, start)
    if j == -1:
        return text[start:].strip()
    return text[start:j].strip()


def make_chat_id(now: Optional[datetime] = None) -> str:
    return (now or datetime.now()).strftime("%m%d-%H%M%S")


def resolve_prompt(prompt: Optional[str], prompt_file: Optional[str | Path]) -> str:
    if prompt_file is not None:
        p = Path(prompt_file)
        return p.read_text(encoding="utf-8")
    if prompt is not None:
        return prompt
    raise ValueError("必须提供 prompt 文本或 prompt_file")


def extract_last_response(text: str) -> str:
    """最后一个 Hermes 框内：顶栏下一行到底栏上一行。"""
    lines = text.splitlines()
    top_idxs = [i for i, ln in enumerate(lines) if HERMES_BOX_TOP in ln]
    bottom_idxs = [i for i, ln in enumerate(lines) if HERMES_BOX_BOTTOM in ln]
    if not top_idxs or not bottom_idxs:
        return ""
    start = top_idxs[-1]
    after = [i for i in bottom_idxs if i > start]
    if not after:
        return ""
    end = after[-1]
    return "\n".join(lines[start + 1 : end])


def extract_session_summary(text: str) -> str:
    """最后一个「Resume this session with:」起至文件末尾。"""
    lines = text.splitlines()
    resume_idxs = [i for i, ln in enumerate(lines) if RESUME_PREFIX in ln]
    if not resume_idxs:
        return ""
    r = resume_idxs[-1]
    return "\n".join(lines[r:])


def parse_hermes_raw_output(text: str) -> tuple[str, str, str, str]:
    return (
        extract_model(text),
        extract_query(text),
        extract_last_response(text),
        extract_session_summary(text),
    )


OUTPUT_SECTION_MODEL = "=== model ==="
OUTPUT_SECTION_QUERY = "=== query ==="
OUTPUT_SECTION_LAST_RESPONSE = "=== last_response ==="
OUTPUT_SECTION_SESSION = "=== session_summary ==="


def format_parsed_output(model: str, query: str, last_response: str, session_summary: str) -> str:
    """四段固定顺序、固定标题，段与段之间空一行。"""
    parts = [
        OUTPUT_SECTION_MODEL,
        model,
        "",
        OUTPUT_SECTION_QUERY,
        query,
        "",
        OUTPUT_SECTION_LAST_RESPONSE,
        last_response,
        "",
        OUTPUT_SECTION_SESSION,
        session_summary,
    ]
    return "\n".join(parts) + "\n"


@dataclass
class HermesChatResult:
    chat_id: str
    temp_output_path: Path
    output_path: Path
    model: str
    query: str
    last_response: str
    session_summary: str
    returncode: int


def hermes_chat(
    prompt: Optional[str] = None,
    *,
    prompt_file: Optional[str | Path] = None,
    timeout: float = 600.0,
    cwd: Optional[str | Path] = None,
    chat_id: Optional[str] = None,
    base_dir: Optional[str | Path] = None,
    tee_output: bool = False,
    login_shell: bool = False,
) -> HermesChatResult:
    """
    在 base_dir（默认 ./hermes_output_logs/）下写入 {chat_id}_hermes_output.txt 与 {chat_id}_hermes_f_output.txt。
    prompt 与 prompt_file 二选一（与 resolve_prompt 一致）。
    tee_output 为 True 时，标准输出与标准错误同时写入 temp 文件并在当前终端显示（依赖 tee）。
    login_shell 为 True 时使用 bash -lc；默认 False，使用 bash -c 并继承当前进程 PATH。
    """
    text = resolve_prompt(prompt, prompt_file)
    cid = chat_id or make_chat_id()
    root = Path(base_dir) if base_dir is not None else DEFAULT_OUTPUT_DIR
    root = root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    # 必须为绝对路径：子进程 cwd 默认与 root 相同时，相对重定向会拼成 root/hermes_output_logs/... 导致父目录不存在
    root = root.resolve()
    temp_path = root / f"{cid}_hermes_output.txt"
    out_path = root / f"{cid}_hermes_f_output.txt"

    cwd_path = Path(cwd).expanduser().resolve() if cwd is not None else root

    qtext = shlex.quote(text)
    qtmp = shlex.quote(str(temp_path))
    if tee_output:
        # pipefail：管道退出码反映 hermes，而非 tee
        inner = f"set -o pipefail; hermes chat -q {qtext} 2>&1 | tee {qtmp}"
    else:
        inner = f"hermes chat -q {qtext} > {qtmp}"
    try:
        proc = subprocess.run(
            ["bash", "-lc" if login_shell else "-c", inner],
            cwd=str(cwd_path),
            timeout=timeout,
            text=True,
        )
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc = 124

    raw = temp_path.read_text(encoding="utf-8", errors="replace") if temp_path.exists() else ""
    model, query, last_response, session_summary = parse_hermes_raw_output(raw)
    out_path.write_text(
        format_parsed_output(model, query, last_response, session_summary),
        encoding="utf-8",
    )

    return HermesChatResult(
        chat_id=cid,
        temp_output_path=temp_path,
        output_path=out_path,
        model=model,
        query=query,
        last_response=last_response,
        session_summary=session_summary,
        returncode=rc,
    )


def _cli() -> int:
    parser = argparse.ArgumentParser(description="运行 hermes chat 并解析输出到 {chat_id}_hermes_f_output.txt")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("-q", "--prompt", help="提示文本")
    g.add_argument("-f", "--prompt-file", help="从文件读取提示文本")
    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        default=600.0,
        help="hermes 子进程超时秒数（默认 600）",
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        default=None,
        help="hermes 工作目录（默认与 --output-dir 相同）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="临时文件与结果文件所在目录（默认 ./hermes_output_logs/）",
    )
    parser.add_argument(
        "--chat-id",
        default=None,
        help="指定 chat_id（默认按当前时间 MMdd-HHmmss）",
    )
    parser.add_argument(
        "--tee",
        action="store_true",
        help="将 hermes 运行过程同时打印到终端并写入 *_hermes_output.txt（默认仅写入文件）",
    )
    parser.add_argument(
        "--login-shell",
        action="store_true",
        help="使用 bash -lc 运行 hermes（默认使用 bash -c 并继承当前进程 PATH）",
    )
    args = parser.parse_args()

    try:
        r = hermes_chat(
            prompt=args.prompt,
            prompt_file=args.prompt_file,
            timeout=args.timeout,
            cwd=args.cwd,
            chat_id=args.chat_id,
            base_dir=args.output_dir,
            tee_output=args.tee,
            login_shell=args.login_shell,
        )
    except FileNotFoundError as e:
        fn = getattr(e, "filename", None)
        if fn == "bash" or (isinstance(fn, str) and fn.endswith("bash")):
            print("未找到 bash", file=sys.stderr)
            return 127
        if fn == "hermes" or (isinstance(fn, str) and "hermes" in fn):
            print("未找到 hermes 命令，请确认已安装并在 PATH 中", file=sys.stderr)
            return 127
        print(e, file=sys.stderr)
        return 1
    except Exception as e:
        print(e, file=sys.stderr)
        return 1

    print(f"chat_id={r.chat_id}")
    print(f"temp={r.temp_output_path}")
    print(f"output={r.output_path}")
    print(f"returncode={r.returncode}")
    if r.returncode == 124:
        print("注意：子进程已超时，输出可能不完整", file=sys.stderr)
    return 0 if r.returncode == 0 else r.returncode


if __name__ == "__main__":
    raise SystemExit(_cli())
