#!/usr/bin/env python3
"""
CLI 或函数调用：运行 `opencode run "<prompt>"`，将标准输出/错误重定向到指定文件。
chat_id 格式：MMdd-HHmmss（strftime %m%d-%H%M%S）；默认或未指定 --output 目录形态时文件名为 {chat_id}_opencode_output.txt。
支持 --tee 同时写入文件并在终端打印。
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


def resolve_prompt(prompt: Optional[str], prompt_file: Optional[str | Path]) -> str:
    if prompt_file is not None:
        return Path(prompt_file).read_text(encoding="utf-8")
    if prompt is not None:
        return prompt
    raise ValueError("必须提供 prompt 文本或 prompt_file")


def make_chat_id(now: Optional[datetime] = None) -> str:
    return (now or datetime.now()).strftime("%m%d-%H%M%S")


def resolve_output_path(output: str | Path | None, chat_id: str) -> Path:
    """
    以 / 结尾（或仅表示目录的路径）视为输出目录，文件名为 {chat_id}_opencode_output.txt；
    未指定 output 时同上，写入当前工作目录。
    否则视为完整输出文件路径（含目录+文件名），不使用 chat_id 改写文件名。
    相对路径相对于进程当前工作目录解析。
    """
    base = Path.cwd()
    named = f"{chat_id}_opencode_output.txt"
    if output is None:
        return (base / named).resolve()
    s = str(output).strip()
    if s.endswith("/") or s.endswith(os.sep):
        dirpath = Path(s.rstrip("/" + os.sep)).expanduser()
        if not dirpath.is_absolute():
            dirpath = base / dirpath
        return dirpath.resolve() / named
    p = Path(s).expanduser()
    if not p.is_absolute():
        p = base / p
    return p.resolve()


@dataclass
class OpencodeChatResult:
    """complete_response 与 output_path 指向文件的内容一致（运行结束后读取）。"""

    chat_id: str
    output_path: Path
    complete_response: str
    returncode: int


def opencode_chat(
    prompt: Optional[str] = None,
    *,
    prompt_file: Optional[str | Path] = None,
    cwd: Optional[str | Path] = None,
    chat_id: Optional[str] = None,
    output: str | Path | None = None,
    tee: bool = False,
    timeout: float | None = None,
) -> OpencodeChatResult:
    """
    在 cwd（默认当前工作目录）下执行 ``opencode run``（prompt 经 shell 转义为单个参数），结果写入 resolve_output_path(output, chat_id)。
    prompt 与 prompt_file 二选一。
    chat_id 未传时按当前时间生成（格式同 hermes_chat.make_chat_id）。
    tee 为 True 时使用 tee 同时写入文件并打印到终端。
    返回的 complete_response 为 output_path 文件当前全文（与磁盘内容一致）。
    """
    text = resolve_prompt(prompt, prompt_file)
    cid = chat_id or make_chat_id()
    out_path = resolve_output_path(output, cid)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print("Opencode 将在此输出临时文件：", out_path)

    cwd_path = Path(cwd).expanduser().resolve() if cwd is not None else Path.cwd()
    cwd_path.mkdir(parents=True, exist_ok=True)

    qtext = shlex.quote(text)
    qout = shlex.quote(str(out_path))
    if tee:
        inner = f"set -o pipefail; opencode run {qtext} 2>&1 | tee {qout}"
    else:
        inner = f"opencode run {qtext} > {qout} 2>&1"

    # 使用 bash -c 而非 -lc：登录 shell 常重置 PATH，导致终端里能用的 opencode 在子进程中找不到
    kwargs: dict = {
        "args": ["bash", "-c", inner],
        "cwd": str(cwd_path),
        "text": True,
    }
    if timeout is not None:
        kwargs["timeout"] = timeout

    try:
        proc = subprocess.run(**kwargs)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc = 124

    complete_response = (
        out_path.read_text(encoding="utf-8", errors="replace") if out_path.exists() else ""
    )
    return OpencodeChatResult(
        chat_id=cid,
        output_path=out_path,
        complete_response=complete_response,
        returncode=rc,
    )


def _cli() -> int:
    parser = argparse.ArgumentParser(description="运行 opencode run 并将输出写入文件")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("-q", "--prompt", help="提示文本")
    g.add_argument("-f", "--prompt-file", help="从文件读取提示文本")
    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        default=600.0,
        metavar="SEC",
        help="opencode 子进程超时秒数（未设置则无超时）",
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        default=None,
        help="工作目录（执行 opencode 的位置；默认当前工作目录）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help="以 / 结尾为输出目录（文件名为 {chat_id}_opencode_output.txt），否则为输出文件的完整路径",
    )
    parser.add_argument(
        "--chat-id",
        default=None,
        help="指定 chat_id（默认按当前时间 MMdd-HHmmss）",
    )
    parser.add_argument(
        "--tee",
        action="store_true",
        help="写入文件的同时打印到终端（默认仅写入文件）",
    )
    args = parser.parse_args()

    try:
        r = opencode_chat(
            prompt=args.prompt,
            prompt_file=args.prompt_file,
            cwd=args.cwd,
            chat_id=args.chat_id,
            output=args.output,
            tee=args.tee,
            timeout=args.timeout,
        )
    except FileNotFoundError as e:
        fn = getattr(e, "filename", None)
        if fn == "bash":
            print("未找到 bash", file=sys.stderr)
            return 127
        # 常见误报：subprocess 因 cwd 不存在抛出 FileNotFoundError，filename 路径中含子串 "opencode"
        print(e, file=sys.stderr)
        return 1
    except Exception as e:
        print(e, file=sys.stderr)
        return 1

    print(f"chat_id={r.chat_id}")
    print(f"output={r.output_path}")
    print(f"returncode={r.returncode}")
    if r.returncode == 124:
        print("注意：子进程已超时，输出可能不完整", file=sys.stderr)
    if r.returncode == 127:
        print(
            "提示：127 通常表示子 shell 内未找到命令。"
            "当前使用 `bash -c`，PATH 继承自启动 Python 的进程；"
            "若 `opencode` 仅在 ~/.bashrc 中配置，请把 PATH 写入 ~/.profile / ~/.bash_profile，"
            "或确认 opencode 为真实可执行文件而非仅 alias/函数。",
            file=sys.stderr,
        )
    return 0 if r.returncode == 0 else r.returncode


if __name__ == "__main__":
    raise SystemExit(_cli())
