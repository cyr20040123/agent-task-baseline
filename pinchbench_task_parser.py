from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any


class PinchbenchTaskParser:
    """解析 Pinchbench 风格任务 Markdown，并按 frontmatter 预置工作区文件。"""

    def __init__(
        self,
        *,
        asset_root: Path | None = None,
        cwd_for_assets: Path | None = None,
    ) -> None:
        self.asset_root = asset_root
        self.cwd_for_assets = cwd_for_assets

    def extract_yaml_frontmatter_raw(self, text: str) -> str | None:
        """返回首个 YAML frontmatter 的正文（不含两侧 ``---``），若无则返回 None。"""
        t = text.lstrip("\ufeff").replace("\r\n", "\n")
        if not t.startswith("---"):
            return None
        m = re.match(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", t, flags=re.DOTALL)
        return m.group(1) if m else None

    def parse_frontmatter(self, text: str) -> dict[str, Any]:
        """解析任务 Markdown 顶部的 YAML frontmatter 为字典；无 frontmatter 时返回空字典。"""
        raw = self.extract_yaml_frontmatter_raw(text)
        if raw is None:
            return {}
        try:
            import yaml
        except ImportError as e:  # pragma: no cover
            raise ImportError("需要安装 PyYAML：python3 -m pip install PyYAML") from e
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as e:
            raise ValueError("任务文件 YAML frontmatter 无法解析") from e
        return data if isinstance(data, dict) else {}

    def resolve_asset_root(self) -> Path:
        """
        资源根目录：优先当前工作目录下的 ``assets``，若不存在则尝试 ``asset``。
        """
        if self.asset_root is not None:
            return self.asset_root.expanduser().resolve()

        base = (self.cwd_for_assets or Path.cwd()).expanduser().resolve()
        assets = base / "assets"
        if assets.is_dir():
            return assets.resolve()
        asset = base / "asset"
        if not asset.is_dir():
            raise FileNotFoundError(f"资源根目录不存在: {asset}")
        return asset.resolve()

    def apply_workspace_files_from_markdown(self, workspace: Path, task_md_raw: str) -> None:
        """
        解析任务文件头部 YAML 的 ``workspace_files``，在已创建的 ``workspace`` 下生成初始文件：

        1. ``path`` + ``content``：将 ``content`` 写入 ``workspace / path``（按需创建父目录）。
        2. ``source`` + ``dest``：将资源根下的 ``source`` 文件或目录拷贝到 ``workspace / dest``。

        每条记录须为 ``path+content`` 或 ``source+dest`` 之一，不可混用或缺字段。
        """
        data = self.parse_frontmatter(task_md_raw)
        spec = data.get("workspace_files")
        if spec is None:
            return
        if not isinstance(spec, list):
            raise ValueError("workspace_files 必须是 YAML 列表")

        root = workspace.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        assets = self.resolve_asset_root()

        for idx, item in enumerate(spec):
            if not isinstance(item, dict):
                raise ValueError(f"workspace_files[{idx}] 必须是映射（字典）")
            has_path = "path" in item and item.get("path") is not None
            has_content = "content" in item and item.get("content") is not None
            has_source = "source" in item and item.get("source") is not None
            has_dest = "dest" in item and item.get("dest") is not None

            if has_path and has_content and not (has_source or has_dest):
                rel = self._assert_safe_rel_path(str(item["path"]), "path")
                out = root / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                body = item["content"]
                if not isinstance(body, str):
                    body = str(body)
                out.write_text(body, encoding="utf-8")
                continue

            if has_source and has_dest and not (has_path or has_content):
                src_rel = self._assert_safe_rel_path(str(item["source"]), "source")
                dst_rel = self._assert_safe_rel_path(str(item["dest"]), "dest")
                src = (assets / src_rel).resolve()
                dst = (root / dst_rel).resolve()
                try:
                    src.relative_to(assets)
                except ValueError as e:
                    raise ValueError(f"source 解析后必须位于资源根目录内: {item['source']!r}") from e
                try:
                    dst.relative_to(root)
                except ValueError as e:
                    raise ValueError(f"dest 解析后必须位于 workspace 内: {item['dest']!r}") from e
                if not src.exists():
                    raise FileNotFoundError(f"资源不存在: {src}（workspace_files[{idx}]）")
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src.is_dir():
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
                continue

            raise ValueError(
                f"workspace_files[{idx}] 须为 path+content 或 source+dest，当前键: {sorted(item.keys())}"
            )

    def strip_yaml_frontmatter(self, text: str) -> str:
        """剥离开头的 YAML frontmatter（--- ... ---），避免被 Markdown 解析器当成标题/段落。"""
        t = text.lstrip("\ufeff").replace("\r\n", "\n")
        if not t.startswith("---"):
            return t
        m = re.match(r"^---\s*\n.*?\n---\s*(?:\n|$)", t, flags=re.DOTALL)
        return t[m.end() :] if m else t

    def extract_prompt_from_markdown(self, text: str) -> str:
        """
        用 markdown-it-py 解析，取第一个二级标题「Prompt」与下一个二级标题之间的正文。
        代码围栏内的 ``##`` 不会被视为新章节边界。
        """
        try:
            from markdown_it import MarkdownIt
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "需要安装 markdown-it-py：python3 -m pip install markdown-it-py"
            ) from e

        stripped = self.strip_yaml_frontmatter(text)
        lines = stripped.splitlines(keepends=True)
        md = MarkdownIt()
        tokens = md.parse(stripped)

        prompt_open_idx = None
        for i, tok in enumerate(tokens):
            if tok.type != "heading_open" or tok.tag != "h2":
                continue
            if i + 1 >= len(tokens) or tokens[i + 1].type != "inline":
                continue
            if tokens[i + 1].content.strip() != "Prompt":
                continue
            prompt_open_idx = i
            break

        if prompt_open_idx is None:
            raise ValueError("未找到「## Prompt」章节（二级标题）")

        inline = tokens[prompt_open_idx + 1]
        if not inline.map:
            raise ValueError("无法定位 Prompt 标题范围")
        start_line = inline.map[1]

        next_h2_line = None
        for j in range(prompt_open_idx + 3, len(tokens)):
            t = tokens[j]
            if t.type == "heading_open" and t.tag == "h2" and t.map:
                next_h2_line = t.map[0]
                break

        end_line = next_h2_line if next_h2_line is not None else len(lines)
        body = "".join(lines[start_line:end_line]).strip()
        body = self._strip_trailing_newline_then_yaml_rule(body)
        if not body:
            raise ValueError("「## Prompt」章节为空")
        return body

    def collect_task_md_files(self, path: Path) -> list[Path]:
        p = path.expanduser().resolve()
        if p.is_file():
            if p.suffix.lower() != ".md":
                raise ValueError(f"不是 Markdown 文件: {p}")
            return [p]
        if p.is_dir():
            files = sorted(p.glob("*.md"))
            if not files:
                raise ValueError(f"目录中没有任何 .md 文件: {p}")
            return files
        raise ValueError(f"路径不存在: {p}")

    @staticmethod
    def _assert_safe_rel_path(p: str, field: str) -> Path:
        rel = Path(p)
        if rel.is_absolute():
            raise ValueError(f"{field} 不得为绝对路径: {p!r}")
        if ".." in rel.parts:
            raise ValueError(f"{field} 不得包含 '..': {p!r}")
        return rel

    @staticmethod
    def _strip_trailing_newline_then_yaml_rule(text: str) -> str:
        """若正文以换行 + 单独一行的 ``---`` 结尾，则反复去掉该后缀。"""
        body = text
        while True:
            body = body.rstrip("\r\n")
            if body.endswith("\r\n---"):
                body = body[: -len("\r\n---")]
            elif body.endswith("\n---"):
                body = body[: -len("\n---")]
            else:
                break
        return body
