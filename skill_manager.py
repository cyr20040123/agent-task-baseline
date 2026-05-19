import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
import subprocess
import sys

def load_config(config_path="agent_configs.json"):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {config_path} not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: {config_path} contains invalid JSON.")
        sys.exit(1)

def format_timestamp(ts):
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

def main():
    parser = argparse.ArgumentParser(description="Manage skills for different agents.")
    parser.add_argument("agent_name", help="The name of the agent to manage")
    parser.add_argument("--path", dest="manager_path", help="Path for managing skill backups", default="./skills_backup")
    parser.add_argument("--save", dest="backup_name_save", metavar="BACKUP_NAME", help="Backup all skills to <manager_path>/<agent_name>/<backup_name>")
    parser.add_argument("--load", dest="backup_name_load", metavar="BACKUP_NAME", help="Load all skills from <manager_path>/<agent_name>/<backup_name>")
    parser.add_argument("--remove-all", action="store_true", help="Remove all existing skills in the agent's environment")

    args = parser.parse_args()

    config = load_config()
    if args.agent_name not in config:
        print(f"Error: Agent '{args.agent_name}' not found in agent_configs.json.")
        sys.exit(1)

    agent_config = config[args.agent_name]
    if "skill_path" not in agent_config:
        print(f"Error: No 'skill_path' configured for agent '{args.agent_name}'.")
        sys.exit(1)

    # Resolve paths (expanding ~)
    skill_path = Path(agent_config["skill_path"]).expanduser().resolve()
    manager_path = Path(args.manager_path).resolve()

    # Create skill path if it doesn't exist to prevent errors
    if not skill_path.exists():
        skill_path.mkdir(parents=True, exist_ok=True)
        print(f"Created skill directory: {skill_path}")

    # --save operation
    if args.backup_name_save:
        backup_dir = manager_path / args.agent_name / args.backup_name_save
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        for item in skill_path.iterdir():
            dest = backup_dir / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
        print(f"Successfully saved skills to backup: {backup_dir}")

    # --load operation
    if args.backup_name_load:
        backup_dir = manager_path / args.agent_name / args.backup_name_load
        if not backup_dir.exists():
            print(f"Error: Backup directory {backup_dir} does not exist.")
        else:
            for item in backup_dir.iterdir():
                dest = skill_path / item.name
                if item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
            print(f"Successfully loaded skills from backup: {backup_dir}")

    # --remove-all operation
    if args.remove_all:
        print(f"\nEvaluating --remove-all for skills in: {skill_path}")
        items = list(skill_path.iterdir())
        
        if not items:
            print("The skill directory is already empty.")
            return

        print("\nExisting skills:")
        for item in items:
            mtime = item.stat().st_mtime
            item_type = "DIR " if item.is_dir() else "FILE"
            print(f" - [{item_type}] {item.name:<30} (Modified: {format_timestamp(mtime)})")
        
        confirmation = input(f"\nAre you sure you want to completely clear '{skill_path}'? [y/N]: ")
        if confirmation.strip().lower() == 'y':
            for item in items:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            print("All skills have been successfully removed.")
            if args.agent_name == "openclaw":
                workspace_path = skill_path.parent
                confirmation2 = input(f"\nAre you sure you want to reset workspace '{workspace_path}'? [y/N]: ")
                if confirmation2.strip().lower() == 'y':
                    cmd_reset = f"openclaw agents delete pinchbench"
                    subprocess.run(cmd_reset, shell=True, check=False)  # Ignore errors if agent doesn't exist
                    if workspace_path.exists():
                        shutil.rmtree(workspace_path)
                    print(f"Workspace '{workspace_path}' has been reset.")
                else:
                    print("Workspace reset aborted.")
        else:
            print("Operation aborted. No skills were removed.")

if __name__ == "__main__":
    main()
