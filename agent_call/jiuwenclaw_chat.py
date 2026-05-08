#!/usr/bin/env python3
"""
CLI 或函数调用：调用 jiuwenclaw_interact.interact_with_jiuwenclaw，并将结果写入指定文件。
chat_id 格式：MMdd-HHmmss（strftime %m%d-%H%M%S）；默认或未指定 --output 目录形态时文件名为 {chat_id}_jiuwenclaw_output.txt。
支持 --tee 将 jiuwenclaw-tui 交互过程同时打印到终端。
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from jiuwenclaw_interact import interact_with_jiuwenclaw  # type: ignore[import-not-found]
except ImportError:
    from .jiuwenclaw_interact import interact_with_jiuwenclaw  # type: ignore[import-not-found]


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
    以 / 结尾（或仅表示目录的路径）视为输出目录，文件名为 {chat_id}_jiuwenclaw_output.txt；
    未指定 output 时同上，写入当前工作目录。
    否则视为完整输出文件路径（含目录+文件名），不使用 chat_id 改写文件名。
    相对路径相对于进程当前工作目录解析。
    """
    base = Path.cwd()
    named = f"{chat_id}_jiuwenclaw_output.txt"
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
class JiuwenClawChatResult:
    """complete_response 与 output_path 指向文件的内容一致（运行结束后读取）。"""

    chat_id: str
    output_path: Path
    complete_response: str
    returncode: int


def jiuwenclaw_chat(
    prompt: Optional[str] = None,
    *,
    prompt_file: Optional[str | Path] = None,
    cwd: Optional[str | Path] = None,
    chat_id: Optional[str] = None,
    output: str | Path | None = None,
    tee: bool = False,
    timeout: float = 600.0,
    command: str = "jiuwenclaw-tui",
    reset_workspace: bool = False,
) -> JiuwenClawChatResult:
    """
    解析 prompt 和输出路径后，直接委托给 jiuwenclaw_interact.interact_with_jiuwenclaw。
    prompt 与 prompt_file 二选一。
    返回的 complete_response 为 output_path 文件当前全文（与磁盘内容一致）。
    """
    text = resolve_prompt(prompt, prompt_file)
    cid = chat_id or make_chat_id()
    out_path = resolve_output_path(output, cid)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cwd_path = Path(cwd).expanduser().resolve() if cwd is not None else Path.cwd()
    cwd_path.mkdir(parents=True, exist_ok=True)

    result = interact_with_jiuwenclaw(
        text,
        interaction_log_file=out_path,
        cwd=cwd_path,
        timeout=timeout,
        tee=tee,
        session_id=cid,
        command=command,
        reset_workspace=reset_workspace,
    )
    complete_response = (
        out_path.read_text(encoding="utf-8", errors="replace")
        if out_path.exists()
        else (result or "")
    )
    returncode = 0 if result is not None else 124

    return JiuwenClawChatResult(
        chat_id=cid,
        output_path=out_path,
        complete_response=complete_response,
        returncode=returncode,
    )


def _cli() -> int:
    parser = argparse.ArgumentParser(description="调用 jiuwenclaw_interact 并将输出写入文件")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("-q", "--prompt", help="提示文本")
    g.add_argument("-f", "--prompt-file", help="从文件读取提示文本")
    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        default=600.0,
        metavar="SEC",
        help="jiuwenclaw-tui 总等待上限秒数（默认 600）",
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        default=None,
        help="工作目录（执行 jiuwenclaw-tui 的位置；默认当前工作目录）",
    )
    parser.add_argument(
        "--reset-workspace",
        action="store_true",
        help="如打开则切换 jiuwenclaw workspace 到 cwd 目录，触发 workspace 重置",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help="以 / 结尾为输出目录（文件名为 {chat_id}_jiuwenclaw_output.txt），否则为输出文件的完整路径",
    )
    parser.add_argument(
        "--chat-id",
        default=None,
        help="指定 chat_id（默认按当前时间 MMdd-HHmmss）",
    )
    parser.add_argument(
        "--command",
        default="jiuwenclaw-tui",
        help="指定 jiuwenclaw TUI 命令（默认 jiuwenclaw-tui）",
    )
    parser.add_argument(
        "--tee",
        action="store_true",
        help="写入文件的同时打印到终端（默认仅写入文件）",
    )
    args = parser.parse_args()

    try:
        r = jiuwenclaw_chat(
            prompt=args.prompt,
            prompt_file=args.prompt_file,
            cwd=args.cwd,
            chat_id=args.chat_id,
            output=args.output,
            tee=args.tee,
            timeout=args.timeout,
            command=args.command,
            reset_workspace=args.reset_workspace,
        )
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 127
    except Exception as e:
        print(e, file=sys.stderr)
        return 1

    print(f"chat_id={r.chat_id}")
    print(f"output={r.output_path}")
    print(f"returncode={r.returncode}")
    if r.returncode == 124:
        print("注意：子进程已超时，输出可能不完整", file=sys.stderr)
    return 0 if r.returncode == 0 else r.returncode


if __name__ == "__main__":
    raise SystemExit(_cli())
