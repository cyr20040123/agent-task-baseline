# agent-task-baseline

## Jiuwenclaw 复用与调用方式

当前 jiuwenclaw 暂无独立的 CLI 调用或 API接口 实现。本项目提供了 CLI + Python 函数的调用实现，如需在别的项目中复用，请直接使用 [agent_call/jiuwenclaw_chat.py](agent_call/jiuwenclaw_chat.py) （需要包含 [agent_call/jiuwenclaw_interact.py](agent_call/jiuwenclaw_interact.py)）。

支持两种调用模式：

1. Python 函数调用模式：在代码中导入并调用对应函数以完成对话或任务处理。

示例：

```python
from agent_call.jiuwenclaw_chat import jiuwenclaw_chat

result = jiuwenclaw_chat(
	prompt="请总结会议要点，并输出为要点列表"
)

# print(result.output_path) # 目前只实现了agent调用，返回格式为TUI流式输出结果不可直接复用
```

2. CLI 调用模式：将该文件作为脚本入口使用，由你的项目侧提供参数解析与命令行封装。

示例：

```bash
python -m agent_call.jiuwenclaw_chat \
	--prompt "请总结会议要点，并输出为要点列表" \
    --workspace "/path/to/workspace/"
	--tee
```

**请注意：目前只实现了 JiuwenClaw 的调用，agent 响应结果无法直接获取**
