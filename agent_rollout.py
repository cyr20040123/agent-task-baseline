#!/usr/bin/env python3
"""
Agent Rollout —— 将 agent_chat 与 llm_proxy_chatml 后台服务封装为一个类，
支持：启动/维护 proxy 进程、初始化 agent 与 workspace、重置 session、
调用 agent_chat、读取 proxy 收集的 ChatML session 输出。

用法：
    python agent_rollout.py -a jiuwenclaw -o ./runs -p "your prompt" --tee

也可作为库使用：
    from agent_rollout import AgentRollout
    with AgentRollout(agent="jiuwenclaw", workspace_dir=Path("./runs")) as rollout:
        result = rollout.execute(prompt="...")
        print(result.chatml_sessions)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from urllib.request import Request, urlopen

from agent_call.jiuwenclaw_chat import jiuwenclaw_chat
from agent_call.openclaw_chat import openclaw_chat, openclaw_init_agent
from agent_call.opencode_chat import opencode_chat

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
AGENT_CONFIGS_PATH = Path(__file__).resolve().with_name("agent_configs.json")
CHANGE_WORKSPACE_PROMPT = (
    'First of all, change current path to this absolute path: "<workspace_path>/", '
    "read and write task-related files only in this directory (except for invoking "
    "the skills)! The file(s) mentioned are only in this folder, if any file is not "
    'found, make sure you are in "<workspace_path>/"!!!\n'
)
DEFAULT_PROXY_INI = Path(__file__).resolve().parent / "llm_proxy_chatml" / "llm_proxy.ini"
DEFAULT_PROXY_SCRIPT = Path(__file__).resolve().parent / "llm_proxy_chatml" / "llm_proxy.py"

logger = logging.getLogger("agent_rollout")

# ---------------------------------------------------------------------------
# 工具函数（保持向后兼容）
# ---------------------------------------------------------------------------


def reset_session_name(url: str, new_name: str, session_path: Optional[str] = None) -> None:
    """向 llm_proxy 发送 /newsession 请求，切换 session 名称和输出路径。"""
    if session_path:
        payload = json.dumps({
            "session_name": new_name,
            "session_path": session_path,
        }).encode("utf-8")
    else:
        payload = json.dumps({"session_name": new_name}).encode("utf-8")
    req = Request(
        f"{url}/newsession",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            logger.info("reset_session_name: %s", result)
    except Exception as e:
        print(f"Error occurred while resetting session name: {e}", file=sys.stderr)


def load_agent_skill_path(agent_name: str, config_path: Path = AGENT_CONFIGS_PATH) -> str:
    """从 agent_configs.json 读取指定 agent 的 skill_path。"""
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"agent_configs.json 无法解析: {config_path}") from e
    if not isinstance(data, dict):
        raise ValueError(f"agent_configs.json 顶层必须是对象: {config_path}")
    entry = data.get(agent_name)
    if not isinstance(entry, dict):
        raise ValueError(f"agent_configs.json 缺少 agent 配置: {agent_name}")
    skill_path = entry.get("skill_path")
    if not isinstance(skill_path, str) or not skill_path:
        raise ValueError(f"agent_configs.json 缺少 skill_path: {agent_name}")
    if skill_path.endswith("/"):
        skill_path = skill_path[:-1]
    return skill_path


def init_agent(agent_name: str, output_dir: Path) -> Path:
    """初始化 agent workspace，返回解析后的输出根目录。"""
    skill_path = load_agent_skill_path(agent_name)
    if agent_name == "openclaw":
        # openclaw比较特殊，只能在workspace中写入，所以要调整output_dir的路径，先构造一个新的workspace，再将output_dir放在workspace下
        print("设置 openclaw workspace ...")
        workspace_root = Path(skill_path).parent
        openclaw_init_agent(
            agent_name="pinchbench",
            workspace_path=str(workspace_root),
            reset=False,
        )
        # 如果args.output_dir是相对路径，则以workspace_root为基准，如果是绝对路径，则将其加在workspace_root后
        if output_dir.is_absolute():
            out_root = workspace_root / output_dir.relative_to("/")
        elif output_dir.is_relative_to(Path.cwd()):
            out_root = workspace_root / output_dir.relative_to(Path.cwd())
        else:
            out_root = workspace_root / output_dir
            print(
                f"Warning: output_dir {output_dir} is not absolute and not relative "
                f"to current working directory; using it as relative to workspace root",
                file=sys.stderr,
            )
            print(f"Resolved output root: {out_root}", file=sys.stderr)
        out_root = out_root.expanduser().resolve()
    else:
        # 而jiuwenclaw则不受限制，直接使用output_dir即可
        out_root = output_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    return out_root


def run_agent_chat(
    *,
    agent: str,
    prompt: str,
    run_chat_id: str,
    timeout: float,
    cwd: Path,
    output: str,
    tee: bool,
    reset_workspace: bool,
    dry_run: bool = False,
    agent_name: str = "pinchbench",
) -> Any:
    """执行一次 agent 对话，返回包含 chat_id / output_path / complete_response / returncode 的结果。"""
    if dry_run:
        print(f"[Dry Run] agent={agent} chat_id={run_chat_id} output={output}")
        print("[Dry Run] -------------- prompt ----------------")
        print(prompt)
        return SimpleNamespace(
            chat_id=run_chat_id,
            output_path=Path(output),
            complete_response="",
            returncode=0,
        )
    if agent == "jiuwenclaw":
        return jiuwenclaw_chat(
            prompt=prompt,
            chat_id=run_chat_id,
            timeout=timeout,
            cwd=cwd,
            output=output,
            tee=tee,
            reset_workspace=reset_workspace,
        )
    elif agent == "opencode":
        return opencode_chat(
            prompt=prompt,
            cwd=cwd,
            timeout=timeout,
            output=output,
            tee=tee,
        )
    return openclaw_chat(
        prompt=prompt,
        chat_id=run_chat_id,
        agent=agent_name,
        timeout=timeout,
        cwd=cwd,
        output=output,
        tee=tee,
    )


# ---------------------------------------------------------------------------
# ChatML 工具函数
# ---------------------------------------------------------------------------


def split_chatml_turns(session: dict) -> list[dict]:
    """
    将一个多轮 ChatML session dict 拆分为逐轮累积的 dict 列表。

    对于 session 中的 *n* 个 assistant 消息，返回 *n* 个 dict，第 *i* 个 dict
    包含从开头到第 *i* 个 assistant（含）的所有消息，以及完整的 system 消息和
    tools 定义。remarks 不会被保留。

    示例
    ----
    输入 3 轮对话（system + 6 个 user/assistant 交替消息）::

        {
            "messages": [sys, u1, a1, u2, a2, u3, a3],
            "tools": [...],
            "remarks": {"incomplete": false}
        }

    输出 3 个 dict::

        [
            {"messages": [sys, u1, a1],        "tools": [...]},   # 第 1 轮
            {"messages": [sys, u1, a1, u2, a2], "tools": [...]},   # 第 1-2 轮
            {"messages": [sys, u1, a1, u2, a2, u3, a3], "tools": [...]},  # 第 1-3 轮
        ]

    参数
    ----
    session : dict
        包含 ``messages`` 和可选的 ``tools`` / ``remarks`` 字段的 ChatML dict。

    返回
    ----
    list[dict]
    """
    messages: list[dict] = session.get("messages", [])
    tools: list[dict] = session.get("tools", [])

    # 找到所有 assistant 消息的位置
    assistant_indices = [
        i for i, m in enumerate(messages) if m.get("role") == "assistant"
    ]

    if not assistant_indices:
        return []

    results: list[dict] = []
    for end_idx in assistant_indices:
        results.append({
            "messages": list(messages[: end_idx + 1]),
            "tools": list(tools),
        })

    return results


# ---------------------------------------------------------------------------
# 结果类型
# ---------------------------------------------------------------------------


@dataclass
class RolloutResult:
    """一次 agent rollout 的完整结果。"""

    chat_id: str
    """agent_chat 的 chat_id"""
    output_path: Path
    """agent_chat 输出文件路径"""
    complete_response: str
    """agent_chat 完整输出文本"""
    returncode: int
    """agent_chat 子进程返回码"""
    chatml_sessions: list[dict] = field(default_factory=list)
    """proxy 收集的 ChatML session 列表"""
    chatml_files: list[Path] = field(default_factory=list)
    """proxy 输出的 ChatML JSON 文件路径列表"""
    session_name: str = ""
    """本次 rollout 的 session name"""
    session_path: str = ""
    """本次 rollout 的 session 输出路径"""


# ---------------------------------------------------------------------------
# AgentRollout
# ---------------------------------------------------------------------------


class AgentRollout:
    """
    管理 llm_proxy_chatml 后台服务生命周期并执行 agent rollout。

    使用方式::

        # 方式 1：手动管理
        rollout = AgentRollout(agent="jiuwenclaw", workspace_dir=Path("./runs"))
        result = rollout.execute(prompt="do something")
        print(result.chatml_sessions)
        rollout.shutdown()

        # 方式 2：上下文管理器（推荐）
        with AgentRollout(agent="jiuwenclaw", workspace_dir=Path("./runs")) as rollout:
            result = rollout.execute(prompt="do something")
    """

    def __init__(
        self,
        *,
        agent: str = "jiuwenclaw",
        workspace_dir: Optional[Path] = None,
        agent_name: str = "pinchbench",
        # proxy 相关
        proxy_ini_path: Optional[Path] = None,
        proxy_script_path: Optional[Path] = None,
        proxy_port: Optional[int] = None,
        proxy_host: str = "127.0.0.1",
        proxy_base_url: Optional[str] = None,
        proxy_api_key: Optional[str] = None,
        proxy_log_folder: Optional[Path] = None,
        proxy_log_chatml: str = "multi",
        proxy_startup_timeout: float = 30.0,
        # agent 初始化
        init_agent_on_start: bool = True,
    ):
        """
        参数
        ----
        agent :
            要使用的 agent 名称（"jiuwenclaw" / "openclaw" / "opencode"）
        workspace_dir :
            运行时目录。若为 None，使用 ``Path.cwd() / "rollout_workspace"``。
        agent_name :
            openclaw 专用 agent 名称（默认 "pinchbench"）
        proxy_ini_path :
            llm_proxy.ini 路径。默认使用 ``llm_proxy_chatml/llm_proxy.ini``。
        proxy_script_path :
            llm_proxy.py 脚本路径。默认使用 ``llm_proxy_chatml/llm_proxy.py``。
        proxy_port :
            proxy 监听端口。None 时使用 ini 中配置的端口。
        proxy_host :
            proxy 监听地址（默认 127.0.0.1）
        proxy_base_url :
            上游 LLM 服务 URL。None 时使用 ini 中的值。
        proxy_api_key :
            上游 API key。None 时使用 ini 中的值。
        proxy_log_folder :
            proxy 日志目录。None 时使用 workspace_dir 下的 logs/。
        proxy_log_chatml :
            ChatML 记录模式："none" / "multi" / "single"（默认 "multi"）
        proxy_startup_timeout :
            proxy 启动等待超时秒数（默认 30）
        init_agent_on_start :
            是否在构造时初始化 agent（默认 True）
        """
        self.agent = agent
        self.agent_name = agent_name
        self.workspace_dir = (
            workspace_dir.expanduser().resolve()
            if workspace_dir is not None
            else Path.cwd() / "rollout_workspace"
        )
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # proxy 配置
        self._proxy_ini_path = proxy_ini_path or DEFAULT_PROXY_INI
        self._proxy_script_path = proxy_script_path or DEFAULT_PROXY_SCRIPT
        self._proxy_port: Optional[int] = proxy_port
        self._proxy_host = proxy_host
        self._proxy_base_url = proxy_base_url
        self._proxy_api_key = proxy_api_key
        self._proxy_log_folder = proxy_log_folder or (self.workspace_dir / "proxy_logs")
        self._proxy_log_chatml = proxy_log_chatml
        self._proxy_startup_timeout = proxy_startup_timeout

        # 运行时状态
        self._proxy_process: Optional[subprocess.Popen] = None
        self._proxy_pid: Optional[int] = None
        self._effective_port: Optional[int] = None
        self._out_root: Optional[Path] = None

        # 读取 ini 获取默认端口
        self._load_proxy_ini_defaults()

        # 启动 proxy
        self._start_proxy()

        # 初始化 agent
        if init_agent_on_start:
            self._out_root = init_agent(self.agent, self.workspace_dir)
        else:
            self._out_root = self.workspace_dir.expanduser().resolve()

    # ------------------------------------------------------------------
    # Proxy 生命周期
    # ------------------------------------------------------------------

    def _load_proxy_ini_defaults(self) -> None:
        """从 llm_proxy.ini 读取默认端口（用于未显式指定 proxy_port 时）。"""
        try:
            import configparser
            ini = configparser.ConfigParser()
            ini.read(str(self._proxy_ini_path))
            # 读取 RECENT 或 DEFAULT 节中的 port
            for section in ("RECENT", "DEFAULT"):
                if ini.has_option(section, "port"):
                    if self._proxy_port is None:
                        self._proxy_port = ini.getint(section, "port")
                if self._proxy_base_url is None and ini.has_option(section, "base-url"):
                    self._proxy_base_url = ini.get(section, "base-url")
                if self._proxy_api_key is None and ini.has_option(section, "api-key"):
                    self._proxy_api_key = ini.get(section, "api-key")
                if ini.has_option(section, "log-folder"):
                    # 仅记下，不覆盖用户指定的 log_folder
                    pass
        except Exception:
            pass

        # 保底默认值
        if self._proxy_port is None:
            self._proxy_port = 8088

    def _start_proxy(self) -> None:
        """启动 llm_proxy 子进程并等待其就绪。"""
        if not self._proxy_script_path.exists():
            raise FileNotFoundError(f"llm_proxy 脚本不存在: {self._proxy_script_path}")

        # 确保 log 目录存在
        self._proxy_log_folder.mkdir(parents=True, exist_ok=True)

        # 构造命令行参数
        cmd = [
            sys.executable,
            str(self._proxy_script_path),
            "--host", self._proxy_host,
            "--port", str(self._proxy_port),
            "--log-folder", str(self._proxy_log_folder),
            "--log-chatml", self._proxy_log_chatml,
        ]
        if self._proxy_base_url:
            cmd += ["--base-url", self._proxy_base_url]
        if self._proxy_api_key:
            cmd += ["--api-key", self._proxy_api_key]

        logger.info("启动 llm_proxy: %s", " ".join(cmd))
        print(f"[AgentRollout] 启动 llm_proxy (端口 {self._proxy_port}) ...")

        # 在 llm_proxy_chatml 目录下启动，以便 configargparse 能找到 llm_proxy.ini
        proxy_cwd = str(self._proxy_script_path.parent)

        self._proxy_process = subprocess.Popen(
            cmd,
            cwd=proxy_cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # 创建新的进程组，便于后续清理
            preexec_fn=os.setsid,
        )
        self._proxy_pid = self._proxy_process.pid
        self._effective_port = self._proxy_port

        # 等待 proxy 就绪
        self._wait_for_proxy(self._proxy_startup_timeout)

    def _wait_for_proxy(self, timeout: float = 30.0) -> None:
        """轮询 /proxyhealth 直到 proxy 响应或超时。"""
        health_url = f"http://{self._proxy_host}:{self._effective_port}/proxyhealth"
        deadline = time.monotonic() + timeout
        last_error = None

        while time.monotonic() < deadline:
            # 检查进程是否还活着
            if self._proxy_process is not None:
                rc = self._proxy_process.poll()
                if rc is not None:
                    stderr = self._proxy_process.stderr.read() if self._proxy_process.stderr else ""
                    raise RuntimeError(
                        f"llm_proxy 进程提前退出 (returncode={rc}):\n{stderr}"
                    )
            try:
                req = Request(health_url)
                with urlopen(req, timeout=2) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read())
                        if data.get("status") == "ok":
                            print(f"[AgentRollout] llm_proxy 就绪 (PID={self._proxy_pid}, "
                                  f"端口={self._effective_port})")
                            return
            except Exception as e:
                last_error = e
            time.sleep(0.5)

        raise TimeoutError(
            f"llm_proxy 在 {timeout}s 内未就绪。最后错误: {last_error}"
        )

    def shutdown(self) -> None:
        """停止 llm_proxy 后台服务。"""
        if self._proxy_process is None:
            return
        print(f"[AgentRollout] 停止 llm_proxy (PID={self._proxy_pid}) ...")
        try:
            # 发送 SIGTERM 到整个进程组（llm_proxy 会 dump session 后退出）
            os.killpg(os.getpgid(self._proxy_process.pid), signal.SIGTERM)
            # 等待优雅退出
            try:
                self._proxy_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # 强制杀死
                os.killpg(os.getpgid(self._proxy_process.pid), signal.SIGKILL)
                self._proxy_process.wait(timeout=5)
        except ProcessLookupError:
            pass  # 已经退出
        except Exception as e:
            logger.warning("停止 proxy 时出错: %s", e)
        finally:
            self._proxy_process = None
            self._proxy_pid = None
            print("[AgentRollout] llm_proxy 已停止")

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def proxy_url(self) -> str:
        """proxy 的 HTTP 根地址。"""
        return f"http://{self._proxy_host}:{self._effective_port}"

    @property
    def proxy_pid(self) -> Optional[int]:
        """proxy 子进程 PID。"""
        return self._proxy_pid

    @property
    def proxy_port(self) -> Optional[int]:
        """proxy 监听端口。"""
        return self._effective_port

    @property
    def out_root(self) -> Optional[Path]:
        """agent 输出根目录。"""
        return self._out_root

    # ------------------------------------------------------------------
    # Session 管理
    # ------------------------------------------------------------------

    def reset_session(self, session_name: str, session_path: Optional[str] = None) -> None:
        """重置 proxy session 名称和输出路径（触发 dump_all）。"""
        reset_session_name(self.proxy_url, session_name, session_path)

    def _iter_chatml_files(self, search_dir: Path):
        """遍历 search_dir 下的 ChatML JSON 文件，自动去重。

        因为 ``*.chatml.json`` 文件同时匹配 ``*.json`` glob，
        所以用 seen set 确保每个文件只 yield 一次。
        """
        seen: set[Path] = set()
        for pattern in ("*.chatml.json", "*.json"):
            for fpath in sorted(search_dir.glob(pattern)):
                if fpath in seen:
                    continue
                seen.add(fpath)
                if fpath.name in ("rollout_summary.json",):
                    continue
                yield fpath

    def read_session_output(self, session_path: Optional[str] = None) -> list[dict]:
        """
        读取 proxy 输出的 ChatML JSON 文件。

        返回一个列表，每个元素是一个 session 的完整数据
        （包含 ``messages`` / ``tools`` / ``remarks`` 等字段）。
        """
        search_dir = Path(session_path) if session_path else self._proxy_log_folder
        if not search_dir.exists():
            logger.warning("session 输出目录不存在: %s", search_dir)
            return []

        sessions: list[dict] = []
        for fpath in self._iter_chatml_files(search_dir):
            try:
                data = json.loads(fpath.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "messages" in data:
                    sessions.append(data)
                elif isinstance(data, list):
                    # single 模式：列表中的每个元素是一个 session
                    for entry in data:
                        if isinstance(entry, dict) and "messages" in entry:
                            sessions.append(entry)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("读取 session 文件失败 %s: %s", fpath, e)

        return sessions

    def find_session_files(self, session_path: Optional[str] = None) -> list[Path]:
        """查找 proxy 输出的 ChatML JSON 文件路径列表。

        只返回包含 ``messages`` 字段的文件（即真正的 ChatML session 输出），
        自动跳过 rollout_summary.json 等其他 JSON 文件。
        """
        search_dir = Path(session_path) if session_path else self._proxy_log_folder
        if not search_dir.exists():
            return []
        files: list[Path] = []
        for fpath in self._iter_chatml_files(search_dir):
            try:
                data = json.loads(fpath.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "messages" in data:
                    files.append(fpath)
                elif isinstance(data, list):
                    if any(isinstance(e, dict) and "messages" in e for e in data):
                        files.append(fpath)
            except (json.JSONDecodeError, OSError):
                pass  # 不是有效的 ChatML JSON，跳过
        return files

    # ------------------------------------------------------------------
    # 执行 rollout
    # ------------------------------------------------------------------

    def execute(
        self,
        *,
        prompt: str,
        run_chat_id: Optional[str] = None,
        timeout: float = 900.0,
        tee: bool = False,
        reset_workspace: bool = True,
        dry_run: bool = False,
        session_name: Optional[str] = None,
        session_path: Optional[str] = None,
        cwd: Optional[Path] = None,
    ) -> RolloutResult:
        """
        执行一次完整的 agent rollout。

        流程：
        1. 初始化 agent（若尚未初始化）
        2. 确定工作目录
        3. 重置 proxy session（触发 dump 上一轮数据）
        4. 调用 agent_chat
        5. 再次重置 session（触发 dump 本轮数据）
        6. 读取并返回 ChatML session 输出

        参数
        ----
        prompt : 提示词文本
        run_chat_id : chat ID（默认按时间生成 MMdd-HHmmss）
        timeout : agent_chat 超时秒数（默认 900）
        tee : 是否同时输出到终端
        reset_workspace : 是否重置 workspace
        dry_run : 仅打印不执行
        session_name : proxy session 名称（默认按时间生成）
        session_path : proxy session 输出路径（默认使用 workspace_dir）
        cwd : agent 工作目录（默认使用 out_root）

        返回
        ----
        RolloutResult
        """
        # 生成默认值
        when = datetime.now()
        run_chat_id = run_chat_id or when.strftime("%m%d-%H%M%S")
        session_name = session_name or f"rollout_{when.strftime('%m%d_%H%M%S')}"
        if cwd is None:
            cwd = self._out_root or self.workspace_dir
        cwd = Path(cwd).expanduser().resolve()
        cwd.mkdir(parents=True, exist_ok=True)

        if session_path is None:
            session_path = str(cwd)
        else:
            session_path = str(Path(session_path).expanduser().resolve())

        # Step 1: 重置 session（dump 之前的数据并开始新的 session）
        print(f"[AgentRollout] 重置 session: {session_name} -> {session_path}")
        self.reset_session(session_name, session_path)

        # Step 2: 构造 output 路径（目录形式，以 / 结尾）
        output_spec = str(cwd) + "/"

        # Step 3: 执行 agent_chat
        print(f"[AgentRollout] 执行 agent_chat (chat_id={run_chat_id}) ...")
        result = run_agent_chat(
            agent=self.agent,
            prompt=prompt,
            run_chat_id=run_chat_id,
            timeout=timeout,
            cwd=cwd,
            output=output_spec,
            tee=tee,
            reset_workspace=reset_workspace,
            dry_run=dry_run,
            agent_name=self.agent_name,
        )

        # Step 4: 再次重置 session 以触发 dump 本轮对话
        if not dry_run:
            time.sleep(0.5)  # 给 proxy 一点时间处理最后的请求
            self.reset_session(
                f"{session_name}_done",
                session_path,
            )

        # Step 5: 读取 ChatML session 输出
        chatml_files = self.find_session_files(session_path)
        chatml_sessions = self.read_session_output(session_path)

        print(f"[AgentRollout] 完成: chat_id={result.chat_id}, "
              f"returncode={result.returncode}, "
              f"chatml_files={len(chatml_files)}, "
              f"chatml_sessions={len(chatml_sessions)}")

        return RolloutResult(
            chat_id=result.chat_id,
            output_path=result.output_path,
            complete_response=result.complete_response,
            returncode=result.returncode,
            chatml_sessions=chatml_sessions,
            chatml_files=chatml_files,
            session_name=session_name,
            session_path=session_path,
        )

    # ------------------------------------------------------------------
    # 上下文管理器
    # ------------------------------------------------------------------

    def __enter__(self) -> "AgentRollout":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()
        return None  # 不抑制异常

    def __del__(self) -> None:
        """析构时尝试清理（不依赖此机制，推荐显式调用 shutdown 或使用 with）。"""
        try:
            self.shutdown()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agent Rollout —— 启动 llm_proxy 后台服务并执行 agent 对话"
    )
    parser.add_argument(
        "-a", "--agent",
        type=str,
        default="jiuwenclaw",
        help="要使用的智能体名称（必须在 agent_configs.json 中配置）",
    )
    parser.add_argument(
        "-o", "--workspace-dir",
        type=Path,
        default=Path.cwd() / "rollout_workspace",
        help="运行时目录（默认 ./rollout_workspace）",
    )
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        required=True,
        help="初始提示词文本，或以 .txt / .md 结尾的文本文件路径",
    )
    parser.add_argument(
        "--tee",
        action="store_true",
        help="是否同时输出到控制台和日志文件",
    )
    parser.add_argument(
        "-t", "--timeout",
        type=float,
        default=900.0,
        help="agent 子进程超时秒数（默认 900）",
    )
    parser.add_argument(
        "--agent-name",
        type=str,
        default="pinchbench",
        help="openclaw 专用 agent 名称（默认 pinchbench）",
    )
    parser.add_argument(
        "--no-init-agent",
        action="store_true",
        help="跳过 agent 初始化",
    )
    parser.add_argument(
        "--no-reset-workspace",
        action="store_true",
        help="不重置 workspace",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印提示词，不实际执行",
    )
    parser.add_argument(
        "--proxy-port",
        type=int,
        default=None,
        help="llm_proxy 监听端口（默认从 llm_proxy.ini 读取）",
    )
    parser.add_argument(
        "--proxy-base-url",
        type=str,
        default=None,
        help="上游 LLM 服务地址（默认从 llm_proxy.ini 读取）",
    )
    parser.add_argument(
        "--proxy-api-key",
        type=str,
        default=None,
        help="上游 API key（默认从 llm_proxy.ini 读取）",
    )
    parser.add_argument(
        "--proxy-log-chatml",
        choices=("none", "multi", "single"),
        default="multi",
        help="ChatML 记录模式（默认 multi）",
    )
    parser.add_argument(
        "--chat-id",
        type=str,
        default=None,
        help="指定 chat_id（默认按时间生成 MMdd-HHmmss）",
    )
    parser.add_argument(
        "--session-name",
        type=str,
        default=None,
        help="指定 proxy session 名称（默认按时间生成）",
    )
    return parser.parse_args()


def _resolve_prompt_arg(prompt_arg: str) -> str:
    """如果 prompt 参数指向一个存在的文件，读取其内容；否则直接返回文本。"""
    p = Path(prompt_arg)
    if p.exists() and p.suffix in (".txt", ".md"):
        return p.read_text(encoding="utf-8")
    return prompt_arg


if __name__ == "__main__":
    args = parse_args()

    # 解析 prompt
    prompt_text = _resolve_prompt_arg(args.prompt)

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 创建 AgentRollout 并执行
    try:
        with AgentRollout(
            agent=args.agent,
            workspace_dir=args.workspace_dir,
            agent_name=args.agent_name,
            proxy_port=args.proxy_port,
            proxy_base_url=args.proxy_base_url,
            proxy_api_key=args.proxy_api_key,
            proxy_log_chatml=args.proxy_log_chatml,
            init_agent_on_start=not args.no_init_agent,
        ) as rollout:
            result = rollout.execute(
                prompt=prompt_text,
                run_chat_id=args.chat_id,
                timeout=args.timeout,
                tee=args.tee,
                reset_workspace=not args.no_reset_workspace,
                dry_run=args.dry_run,
                session_name=args.session_name,
            )

            # 输出结果摘要
            print(f"\n{'='*60}")
            print(f"Rollout 完成")
            print(f"  chat_id:       {result.chat_id}")
            print(f"  output_path:   {result.output_path}")
            print(f"  returncode:    {result.returncode}")
            print(f"  session_name:  {result.session_name}")
            print(f"  session_path:  {result.session_path}")
            print(f"  chatml_files:  {len(result.chatml_files)}")
            print(f"  chatml_sessions: {len(result.chatml_sessions)}")
            if result.returncode == 124:
                print(f"  注意：子进程已超时，输出可能不完整", file=sys.stderr)

            # 将 ChatML session 汇总写入 rollout_summary.json
            summary_path = rollout.out_root / "rollout_summary.json" if rollout.out_root else None
            if summary_path:
                summary = {
                    "chat_id": result.chat_id,
                    "output_path": str(result.output_path),
                    "returncode": result.returncode,
                    "session_name": result.session_name,
                    "session_path": result.session_path,
                    "chatml_files": [str(f) for f in result.chatml_files],
                    "num_chatml_sessions": len(result.chatml_sessions),
                    "chatml_sessions": result.chatml_sessions,
                }
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                summary_path.write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"  summary:       {summary_path}")

    except KeyboardInterrupt:
        print("\n中断", file=sys.stderr)
        raise SystemExit(130)
    except Exception as e:
        logger.error("rollout 失败: %s", e, exc_info=True)
        print(f"错误: {e}", file=sys.stderr)
        raise SystemExit(1)
