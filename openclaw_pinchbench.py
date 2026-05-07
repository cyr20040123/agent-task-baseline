#!/usr/bin/env python3
"""
批量运行 Pinchbench 任务：从 Markdown 的「## Prompt」章节提取提示词，
调用 openclaw 或 jiuwenclaw 作为执行 agent；每个任务结束后通过 opencode_chat 做评分
（cwd 为运行本脚本时的当前工作目录）。

用法：
    python openclaw_pinchbench.py <任务.md 或 任务目录> <输出根目录>

每个任务在 <输出根目录>/<文件名去 .md>_<MMdd_HHmm>/ 下执行（agent cwd 与日志目录相同）。
若任务 YAML 中含 ``workspace_files``，会在该目录下按 ``path+content`` 或 ``source+dest``
预置文件（资源默认从当前目录 ``assets`` 或 ``asset`` 读取）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import csv

from agent_call.jiuwenclaw_chat import jiuwenclaw_chat
from agent_call.openclaw_chat import openclaw_chat
from pinchbench_task_parser import PinchbenchTaskParser

CSV_HISTORY_FILE = "pinchbench_exec_history.csv"
CSV_HEADERS = ["timestamp", "command", "agent", "task_score", "full_score", "remarks"]

def run_dir_name(stem: str, when: datetime, *, with_seconds: bool = False) -> str:
    fmt = "%m%d_%H%M%S" if with_seconds else "%m%d_%H%M"
    return f"{stem}_{when.strftime(fmt)}"


def make_chat_id(when: datetime | None = None) -> str:
    return (when or datetime.now()).strftime("%m%d-%H%M%S")


def allocate_run_dir(out_root: Path, stem: str, when: datetime) -> Path:
    """优先使用 MMdd_HHmm；若目录已存在则改用 MMdd_HHmmss，避免同分钟重复运行覆盖。"""
    name = run_dir_name(stem, when, with_seconds=False)
    sub = out_root / name
    if sub.exists():
        name = run_dir_name(stem, when, with_seconds=True)
        sub = out_root / name
    return sub


_SCORE_LINE_RE = re.compile(
    r"TASK_SCORE\s*=\s*(-?[\d.]+)\s*,\s*FULL_SCORE\s*=\s*(-?[\d.]+)",
    re.IGNORECASE,
)

PROMPTS_PATH = Path(__file__).resolve().with_name("prompts.json")
AGENT_CONFIGS_PATH = Path(__file__).resolve().with_name("agent_configs.json")
OPENCODE_EVALUATION_PROMPT_KEY = "opencode_evaluation_prompt_cn"
INVOKE_SKILL_PROMPT_KEY = "invoke_skill_prompt"
NAIVE_SKILL_EVOLUTION_PROMPT_KEY = "naive_skill_evolution_prompt"


def parse_task_score_from_opencode_output(text: str) -> tuple[float | None, float | None]:
    """从 opencode 输出中自末行向上查找 ``TASK_SCORE=..., FULL_SCORE=...``。"""
    for line in reversed(text.strip().splitlines()):
        line_st = line.strip()
        if not line_st:
            continue
        m = _SCORE_LINE_RE.search(line_st)
        if m:
            try:
                return float(m.group(1)), float(m.group(2))
            except ValueError:
                return None, None
    return None, None


def load_prompt_template(
    key: str,
    prompts_path: Path = PROMPTS_PATH,
    replace_dict: dict[str, str] | None = None,
) -> str:
    """从 prompts.json 读取 prompt 模板，避免在脚本中硬编码评测指令。"""
    try:
        data = json.loads(prompts_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"prompts.json 无法解析: {prompts_path}") from e
    if not isinstance(data, dict):
        raise ValueError(f"prompts.json 顶层必须是对象: {prompts_path}")
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"prompts.json 缺少非空字符串键: {key}")
    for k, v in (replace_dict or {}).items():
        k = f"<{k}>" if not k.startswith("<") else k
        value = value.replace(k, v)
    return value


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
    return skill_path


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
) -> Any:
    if dry_run:
        return SimpleNamespace(
            chat_id=run_chat_id,
            output_path=Path(output),
            complete_response="",
            returncode=0
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
    return openclaw_chat(
        prompt=prompt,
        chat_id=run_chat_id,
        session_id=run_chat_id,
        timeout=timeout,
        cwd=cwd,
        output=output,
        tee=tee,
    )


def opencode_evaluation(
    task_description_file: Path | str,
    task_output_dir: Path | str,
    *,
    run_id: str,
    timeout: float | None = 600.0,
    tee: bool = False,
    dry_run: bool = False,
) -> tuple[float | None, float | None, Any]:
    """
    使用 ``opencode_chat`` 调用 opencode 评分；``cwd`` 为当前进程工作目录（运行 ``openclaw_pinchbench.py`` 时的目录）。
    ``task_description_file``、``task_output_dir`` 以绝对路径写入评判提示词。
    评判输出写入 ``task_output_dir / f"{run_id}_opencode_eval.txt"``。
    """
    from agent_call.opencode_chat import opencode_chat

    desc_abs = str(Path(task_description_file).expanduser().resolve())
    out_abs = str(Path(task_output_dir).expanduser().resolve())
    judge_prompt = load_prompt_template(OPENCODE_EVALUATION_PROMPT_KEY, replace_dict={"task_description_file": str(desc_abs), "task_output_dir": str(out_abs)})
    # judge_prompt = (
    #     judge_prompt_cn.replace("<task_description_file>", desc_abs).replace(
    #         "<task_output_dir>", out_abs
    #     )
    # )
    eval_out = Path(task_output_dir).expanduser().resolve() / f"{run_id}_opencode_eval.txt"
    
    if dry_run:
        return 1.0, 1.0, SimpleNamespace(
            chat_id=run_id,
            output_path=eval_out,
            complete_response="",
            returncode=0,
        )
    
    result = opencode_chat(
        prompt=judge_prompt,
        cwd=Path.cwd().resolve(),
        output=str(eval_out),
        timeout=timeout,
        tee=tee,
    )
    task_score, full_score = parse_task_score_from_opencode_output(result.complete_response)
    return task_score, full_score, result


def main() -> int:
    command = " ".join(sys.argv)

    parser = argparse.ArgumentParser(
        description="从任务描述 .md 提取 ## Prompt，逐个调用 openclaw_chat；"
        "每任务完成后用 opencode_chat 评分（cwd 为运行本脚本时的当前工作目录）。"
    )
    parser.add_argument(
        "task",
        type=Path,
        help="单个任务描述 .md 文件，或包含多个 .md 的目录",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=Path.cwd() / "pinchbench_runs",
        help="输出根目录（其下为每个任务的运行子目录）",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        default=600.0,
        help="单次 openclaw 子进程超时秒数（默认 600）",
    )
    parser.add_argument(
        "--agent",
        choices=("openclaw", "jiuwenclaw"),
        default="openclaw",
        help="选择执行任务的 agent（默认 openclaw）",
    )
    parser.add_argument(
        "--tee",
        action="store_true",
        help="openclaw 与 opencode 评分输出同时打印终端并写入对应日志文件",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=None,
        help="source→dest 拷贝时的资源根目录（默认：当前目录下 assets，若无则 asset）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印提示词，不运行 openclaw_chat",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="跳过 opencode 评分环节",
    )
    parser.add_argument(
        "--skill-evolve",
        choices=("none", "naive"),
        default="none",
        help="是否在评分后进行 skill 自进化（默认 none）",
    )
    parser.add_argument(
        "--skill-name",
        default=None,
        help="强制所有任务使用同一 skill_name（默认使用任务文件名）",
    )
    parser.add_argument(
        "--opencode-timeout",
        type=float,
        default=600.0,
        metavar="SEC",
        help="单次 opencode 评分子进程超时秒数（默认 600）",
    )
    args = parser.parse_args()

    out_root = args.output_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    task_parser = PinchbenchTaskParser(
        asset_root=args.asset_root,
        cwd_for_assets=Path.cwd(),
    )

    try:
        md_files = task_parser.collect_task_md_files(args.task)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2

    exit_code = 0
    i = 0
    score_rows: list[tuple[float, float]] = []
    openclaw_runs = 0
    for md_path in md_files:
        i += 1
        stem = md_path.stem
        when = datetime.now()
        sub = allocate_run_dir(out_root, stem, when)
        sub.mkdir(parents=True, exist_ok=True)

        try:
            raw = md_path.read_text(encoding="utf-8")
            task_parser.apply_workspace_files_from_markdown(sub, raw)
            task_prompt = task_parser.extract_prompt_from_markdown(raw)
        except (
            OSError,
            UnicodeError,
            ValueError,
            FileNotFoundError,
            ImportError,
        ) as e:
            print(f"[跳过] {md_path}: {e}", file=sys.stderr)
            exit_code = 1
            continue

        skill_name = args.skill_name or md_path.stem
        prompt = ""
        if args.skill_evolve == "naive":
            prompt += load_prompt_template(INVOKE_SKILL_PROMPT_KEY, replace_dict={"skill_name": skill_name})
        if args.agent == "openclaw":
            prompt += load_prompt_template(
                "change_openclaw_workspace_prompt",
                replace_dict={"workspace_path": str(sub)},
            )
        prompt += task_prompt

        print(f"\n[任务 {i}]: {md_path}")
        print(f"[运行目录] {sub}")

        # if (args.dry_run):
        #     print(f"[提示词] {prompt}")
        #     continue

        run_chat_id = make_chat_id()
        try:
            r = run_agent_chat(
                agent=args.agent,
                prompt=prompt,
                run_chat_id=run_chat_id,
                timeout=args.timeout,
                cwd=sub,
                output=str(sub) + "/",
                tee=args.tee,
                reset_workspace=True,
                dry_run=args.dry_run,
            )
        except Exception as e:
            print(f"[失败] {md_path}: {e}", file=sys.stderr)
            exit_code = 1
            continue

        print(f"chat_id={r.chat_id} returncode={r.returncode}")
        # if args.agent == "openclaw":
        #     print(f"session_id={r.get('session_id', '')}")
        print(f"output={r.output_path}")
        if r.returncode != 0:
            exit_code = 1
            if r.returncode == 124:
                print("注意：子进程已超时，输出可能不完整", file=sys.stderr)

        openclaw_runs += 1
        if (not args.skip_eval):
            print("[评分] 调用 opencode …")
            try:
                task_score, full_score, ocr = opencode_evaluation(
                    md_path.resolve(),
                    sub,
                    run_id=r.chat_id,
                    timeout=args.opencode_timeout,
                    tee=args.tee,
                    dry_run=args.dry_run,
                )
            except Exception as e:
                print(f"[评分失败] {md_path}: {e}", file=sys.stderr)
                exit_code = 1
                continue
            print(f"[评分] opencode chat_id={ocr.chat_id} returncode={ocr.returncode}")
            print(f"[评分] 输出文件={ocr.output_path}")
            if ocr.returncode != 0:
                exit_code = 1
                if ocr.returncode == 124:
                    print("注意：opencode 子进程已超时，评分输出可能不完整", file=sys.stderr)
            if task_score is not None and full_score is not None:
                score_rows.append((task_score, full_score))
                print(f"[评分] TASK_SCORE={task_score} FULL_SCORE={full_score}")
            else:
                print(
                    "[评分] 未能从 opencode 输出中解析 TASK_SCORE=..., FULL_SCORE=...",
                    file=sys.stderr,
                )
                exit_code = 1

            if args.skill_evolve == "naive" and ocr.returncode == 0 and r.returncode == 0:
                try:
                    skill_path = load_agent_skill_path(args.agent)
                    evolve_prompt = load_prompt_template(
                        NAIVE_SKILL_EVOLUTION_PROMPT_KEY,
                        replace_dict={
                            "task_evaluation_file": str(ocr.output_path),
                            "skill_path": skill_path,
                            "skill_name": skill_name,
                        },
                    )
                    evolve_out = sub / f"{run_chat_id}_{args.agent}_evolve_output.txt"
                    _ = run_agent_chat(
                        agent=args.agent,
                        prompt=evolve_prompt,
                        run_chat_id=run_chat_id,
                        timeout=args.timeout,
                        cwd=sub,
                        output=str(evolve_out),
                        tee=args.tee,
                        reset_workspace=False,
                    )
                except Exception as e:
                    print(f"[Skill Evolver 失败] {md_path}: {e}", file=sys.stderr)
                    exit_code = 1
            elif args.skill_evolve == "naive":
                print(f"[Skill Evolver] 跳过进化：仅在 agent 执行成功且 llm judge 评分成功时才进化", file=sys.stderr)

    if not args.skip_eval and score_rows:
        total_s = sum(s for s, _ in score_rows)
        total_f = sum(f for _, f in score_rows)
        print(f"\n[本批评分汇总] 总分 {total_s:g} / 满分 {total_f:g}")
        if openclaw_runs > len(score_rows):
            print(
                f"[本批评分汇总] 提示：共 {openclaw_runs} 次 openclaw 运行，"
                f"仅 {len(score_rows)} 次解析到有效分数。",
                file=sys.stderr,
            )
    elif not args.skip_eval and openclaw_runs > 0:
        print(
            "\n[本批评分汇总] 未能汇总：没有任何任务解析到 TASK_SCORE / FULL_SCORE。",
            file=sys.stderr,
        )
    
    csv_row = {}
    for field in CSV_HEADERS:
        if field == "timestamp":
            csv_row[field] = datetime.now().isoformat()
        elif field == "command":
            csv_row[field] = command
        elif field == "agent":
            csv_row[field] = args.agent
        elif field == "task_score":
            csv_row[field] = total_s if score_rows else ""
        elif field == "full_score":
            csv_row[field] = total_f if score_rows else ""
        elif field == "remarks":
            csv_row[field] = f"{i} tasks, {openclaw_runs} openclaw runs, {len(score_rows)} scored" if openclaw_runs > 0 else ""

    # Write the CSV row to the history file
    with open(CSV_HISTORY_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(csv_row)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

# sample usage:
# python3 openclaw_pinchbench.py selected_tasks/task_polymarket_briefing.md -o ./pinchbench_runs
# python3 openclaw_pinchbench.py selected_tasks -o ./pinchbench_runs --tee
# python3 ./openclaw_pinchbench.py ./lytton_selected_tasks/ -o ./runs/openclaw --tee
