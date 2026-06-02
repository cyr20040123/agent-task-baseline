---
name: TASK-NAME-process-reward
description: Evaluate the trajectory of accomplishing the task "TASK-NAME" by following the scoring point.
---

# 技能描述
输入一个Agent完成 TASK-NAME 任务的历史多轮对话轨迹（可能是执行到一半还未完成的轨迹，也可能是完整轨迹）。基于轨迹数据，同时参考以下得分点描述信息，检查匹配轨迹中是否存在相应的得分点，根据得分点描述对轨迹完成任务情况进行打分，并按规定格式输出。

# 得分点描述
<!-- 请在此填充得分点信息，以表格形式表示，每行一个得分点，每行有三列，第一列为得分点描述（执行特定动作、调用特定工具、获得指定信息等），第二列表示当前得分点的得分标准，第三列表示符合这条得分标准所获得的分数。以下是示例（需要替换）： -->
| 得分点描述 | 得分标准 | 得分 |
| ---- | ---- | ---- |
| 阅读 "transcript.txt"| 用任意方式成功读取文件 | 1 |
| 查找小明的分数 | 找到小明的分数 | 1 |
| 尝试计算了总成绩 | 只要尝试计算总成绩 | 0.5 |
| 计算出正确的总成绩为634分 | 计算出正确的总成绩 | 1 |
| 最终输出了成绩等级 | 在最后一轮响应中输出了成绩等级信息 | 1 |
满分：xx（上述得分点总和）

# 输入格式
一个ChatML格式的执行任务的对话轨迹（可能在上下文中或来自json文件），轨迹可能完成到一半，也可能是完成任务的完整轨迹，包含用户初始提示词、 agent 侧的工具响应、大模型回应等内容。如果输入的是一个文件，先不要读取文件，直接用下面代码的clean_message_from_file函数清理一遍，读取清理后的内容（若提取脚本执行失败则直接读取原始轨迹文件）。

# 干净轨迹提取脚本
```python
"""Print ChatML trajectory messages, stripping system messages and noise fields."""

import json
from pathlib import Path

_EXCLUDE_KEYS = {"timestamp", "prompt_ids", "completion_ids", "logprobs"}
def filter_message(msg: dict) -> dict:
    return {k: v for k, v in msg.items() if k not in _EXCLUDE_KEYS}

def clean_message_from_file(json_file: Path):
    data = json.loads(json_text = file.read_text(encoding="utf-8"))
    messages = data.get("messages", data) if isinstance(data, dict) else data
    return [filter_message(m) for m in messages if m.get("role") != "system"]
    return clean_message(json_text)
```

# 输出格式
（需包含以下输出，下面内容需要严格输出到回应中，如果用户有要求则同时输出到文件中）
输出第1行是固定字符串 [PROCESS-REWARD]
输出第2行是一个数字，表示总得分 S ，即基于上述得分点描述和统计表格，当前轨迹完成了或匹配上的所有的得分点的总分，精确到小数点后两位。
输出第3行是一个数字，表示满分（即所有的得分点分数上限之和） S0，精确到小数点后两位。
输出第4行是一个数字，表示归一化总得分 S1 。S1 = S / S0，精确到小数点后两位。
最后输出一个表格：
表格每行对应得分点描述里的一个得分点，每行有三列信息，分别是得分点描述、当前得分点满分分数、当前得分点得分分数。
