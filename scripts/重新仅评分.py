from agent_call.opencode_chat import opencode_chat
from agent_pinchbench import load_prompt_template, opencode_evaluation, parse_task_score_from_opencode_output

import argparse

def manual_eval(task_description_file, task_output_dir):
    res = opencode_evaluation(task_description_file, task_output_dir, tee = True)

def main(args):
    parser = argparse.ArgumentParser(description="Run manual evaluation for Pinchbench")
    parser.add_argument("task_description_file", help="Path to the task description file")
    parser.add_argument("runs_dir", help="Path to the task output directory")
    # for each subfolder in runs_dir, run evaluation and save result to a file named {subfolder}_opencode_eval.txt in the same directory
    args = parser.parse_args()
    manual_eval(args.task_description_file, args.task_output_dir)