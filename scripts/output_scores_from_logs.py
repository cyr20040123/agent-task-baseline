"""
执行参数：python output_scores_from_logs.py /path/to/logs
功能：遍历日志目录中的所有子目录，找到结尾为"_eval.txt"的文件，提取其中的分数信息，输出到CLI中。
分数信息如何提取：通过正则表达式匹配文件内容中的"TASK_SCORE="后面的数字，和"FULL_SCORE="后面的数字。注意=左右可能有空格。
输出格式：对于每个找到的_eval.txt文件，输出一行，包含最后一层目录名、提取的TASK_SCORE和FULL_SCORE，格式如下：
<技能名称>: TASK_SCORE = <提取的TASK_SCORE>, FULL_SCORE = <提取的FULL_SCORE>
"""
import re
from pathlib import Path
import sys

if len(sys.argv) != 2:
    print("Usage: python output_scores_from_logs.py /path/to/logs")
    sys.exit(1)

log_dir = Path(sys.argv[1])
if not log_dir.is_dir():
    print(f"Error: {log_dir} is not a valid directory.")
    sys.exit(1)

# 定义正则表达式模式
task_score_pattern = re.compile(r"TASK_SCORE\s*=\s*([0-9.]+)")
full_score_pattern = re.compile(r"FULL_SCORE\s*=\s*([0-9.]+)")

scores = []

# 遍历日志目录中的所有子目录
for eval_file in log_dir.rglob("*_eval.txt"):
    try:
        content = eval_file.read_text()
        
        # 提取分数信息
        task_score_match = task_score_pattern.search(content)
        full_score_match = full_score_pattern.search(content)
        
        if task_score_match and full_score_match:
            task_score = task_score_match.group(1)
            full_score = full_score_match.group(1)
            skill_name = eval_file.parent.name  # 获取最后一层目录名作为技能名称
            print(f"{skill_name}: TASK_SCORE = {task_score}, FULL_SCORE = {full_score}")
            scores.append(task_score)
        else:
            print(f"Warning: Could not find scores in {eval_file}")
    except Exception as e:
        print(f"Error processing {eval_file}: {str(e)}")

print(f"\nExtracted TASK_SCORE values: {len(scores)}")
for score in scores:
    print(score)