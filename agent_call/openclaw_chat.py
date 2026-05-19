#!/usr/bin/env python3
"""
CLI 或函数调用：运行 `openclaw agent --message "<prompt>" --local`，将标准输出/错误重定向到指定文件。
chat_id 格式：MMdd-HHmmss（strftime %m%d-%H%M%S）；默认或未指定 --output 目录形态时文件名为 {chat_id}_openclaw_output.txt。
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


THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh", "adaptive", "max")


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
    以 / 结尾（或仅表示目录的路径）视为输出目录，文件名为 {chat_id}_openclaw_output.txt；
    未指定 output 时同上，写入当前工作目录。
    否则视为完整输出文件路径（含目录+文件名），不使用 chat_id 改写文件名。
    相对路径相对于进程当前工作目录解析。
    """
    base = Path.cwd()
    named = f"{chat_id}_openclaw_output.txt"
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
class OpenClawChatResult:
    """complete_response 与 output_path 指向文件的内容一致（运行结束后读取）。"""

    chat_id: str
    # session_id: str
    output_path: Path
    complete_response: str
    returncode: int


def openclaw_init_agent(agent_name: str = "pinchbench", workspace_path: str = "./pinchbench_runs/openclaw/workspace", reset: bool = False) -> bool:
    cmd_reset = f"openclaw agents delete {agent_name} --force"
    cmd = f"openclaw agents add {agent_name} --workspace {workspace_path}"
    try:
        if reset:
            subprocess.run(cmd_reset, shell=True, check=True)
        subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def openclaw_chat(
    prompt: Optional[str] = None,
    *,
    prompt_file: Optional[str | Path] = None,
    cwd: Optional[str | Path] = None,
    chat_id: Optional[str] = None,
    # session_id: Optional[str] = None,
    agent: str | None = None,
    thinking: str = "medium",
    output: str | Path | None = None,
    tee: bool = False,
    timeout: float | None = None,
    local: bool = True,
) -> OpenClawChatResult:
    """
    在 cwd（默认当前工作目录）下执行 ``openclaw agent``，结果写入 resolve_output_path(output, chat_id)。
    prompt 与 prompt_file 二选一。
    session_id 未传时使用 chat_id。
    thinking 控制推理强度，可选 off/minimal/low/medium/high/xhigh/adaptive/max。
    local 为 True 时追加 --local。
    返回的 complete_response 为 output_path 文件当前全文（与磁盘内容一致）。
    """
    if thinking not in THINKING_LEVELS:
        choices = ", ".join(THINKING_LEVELS)
        raise ValueError(f"thinking 必须是以下之一：{choices}")

    prompt_text = resolve_prompt(prompt, prompt_file)
    cid = chat_id or make_chat_id()
    # sid = session_id or cid
    out_path = resolve_output_path(output, cid)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cwd_path = Path(cwd).expanduser().resolve() if cwd is not None else Path.cwd()
    cwd_path.mkdir(parents=True, exist_ok=True)

    if agent is not None and len(agent.strip()) > 0:
        qagent = f"--agent {shlex.quote(agent)}"
    else:
        qagent = "" # f"{shlex.quote(cid)}"  # 复用 chat_id 作为 agent 名称，避免用户还要额外指定一个参数
    # qsid = shlex.quote(sid)
    qsid = shlex.quote(cid)
    qthinking = shlex.quote(thinking)
    qprompt_text = shlex.quote(prompt_text)
    qout = shlex.quote(str(out_path))
    cmd = (
        f"openclaw agent {qagent} --session-id {qsid} "
        f"--thinking {qthinking} --message {qprompt_text}"
    )
    if local:
        cmd = f"{cmd} --local"
    if tee:
        inner = f"set -o pipefail; {cmd} 2>&1 | tee {qout}"
    else:
        inner = f"{cmd} > {qout} 2>&1"
    
    # input(f"Running command in {cwd_path}:\n{inner}")
    
    # 使用 bash -c 而非 -lc：登录 shell 常重置 PATH，导致终端里能用的 openclaw 在子进程中找不到
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
    return OpenClawChatResult(
        chat_id=cid,
        # session_id=cid,
        output_path=out_path,
        complete_response=complete_response,
        returncode=rc,
    )


def _cli() -> int:
    parser = argparse.ArgumentParser(description="运行 openclaw agent 并将输出写入文件")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("-q", "--prompt", help="提示文本")
    g.add_argument("-f", "--prompt-file", help="从文件读取提示文本")
    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        default=600.0,
        metavar="SEC",
        help="openclaw 子进程超时秒数（未设置则无超时）",
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        default=None,
        help="工作目录（执行 openclaw 的位置；默认当前工作目录）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help="以 / 结尾为输出目录（文件名为 {chat_id}_openclaw_output.txt），否则为输出文件的完整路径",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="指定 openclaw session_id，并作为输出文件名标识（默认按当前时间 MMdd-HHmmss）",
    )
    parser.add_argument(
        "--agent",
        default="main",
        help="指定 openclaw agent（默认 main；也可传 default）",
    )
    parser.add_argument(
        "--thinking",
        choices=THINKING_LEVELS,
        default="medium",
        help="指定 thinking 等级（默认 medium）",
    )
    parser.add_argument(
        "--tee",
        action="store_true",
        help="写入文件的同时打印到终端（默认仅写入文件）",
    )
    parser.add_argument(
        "--no-local",
        action="store_true",
        help="不追加 --local（默认追加）",
    )
    args = parser.parse_args()

    try:
        r = openclaw_chat(
            prompt=args.prompt,
            prompt_file=args.prompt_file,
            cwd=args.cwd,
            chat_id=args.session_id,
            session_id=args.session_id,
            agent=args.agent,
            thinking=args.thinking,
            output=args.output,
            tee=args.tee,
            timeout=args.timeout,
            local=not args.no_local,
        )
    except FileNotFoundError as e:
        fn = getattr(e, "filename", None)
        if fn == "bash":
            print("未找到 bash", file=sys.stderr)
            return 127
        print(e, file=sys.stderr)
        return 1
    except Exception as e:
        print(e, file=sys.stderr)
        return 1

    print(f"chat_id={r.chat_id}")
    print(f"session_id={r.session_id}")
    print(f"output={r.output_path}")
    print(f"returncode={r.returncode}")
    if r.returncode == 124:
        print("注意：子进程已超时，输出可能不完整", file=sys.stderr)
    if r.returncode == 127:
        print(
            "提示：127 通常表示子 shell 内未找到命令。"
            "当前使用 `bash -c`，PATH 继承自启动 Python 的进程；"
            "若 `openclaw` 仅在 ~/.bashrc 中配置，请把 PATH 写入 ~/.profile / ~/.bash_profile，"
            "或确认 openclaw 为真实可执行文件而非仅 alias/函数。",
            file=sys.stderr,
        )
    return 0 if r.returncode == 0 else r.returncode


if __name__ == "__main__":
    raise SystemExit(_cli())