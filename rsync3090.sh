#!/usr/bin/env bash

set -euo pipefail

# ===== Hardcoded sync target =====
#REMOTE_USER="fq9hpsacuser04"
REMOTE_HOST="3090-150"
REMOTE_PORT="22"
REMOTE_DIR="/root/cyr/development/agent-task-baseline"

# Sync current folder (the folder where this script lives)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${SCRIPT_DIR}/"
# DESTINATION="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
DESTINATION="${REMOTE_HOST}:${REMOTE_DIR}/"

# Default behavior: dry-run only (preview)
DRY_RUN=1
DELETE_SYNC=0

for arg in "$@"; do
	case "$arg" in
	--apply|-a)
		DRY_RUN=0
		;;
	--delete)
		DELETE_SYNC=1
		;;
	-h|--help)
		echo "Usage: $0 [--apply|-a] [--delete]"
		echo "  (default)      Dry-run preview"
		echo "  --apply, -a    Execute actual rsync"
		echo "  --delete       Delete remote files not present locally (effective only with --apply/-a)"
		exit 0
		;;
	*)
		echo "Unknown option: $arg" >&2
		echo "Usage: $0 [--apply|-a] [--delete]" >&2
		exit 1
		;;
	esac
done

if [[ ! -f "${SCRIPT_DIR}/.gitignore" ]]; then
	echo "Error: .gitignore not found in ${SCRIPT_DIR}" >&2
	exit 1
fi

RSYNC_OPTS=(
	-azP
	--filter=':- .gitignore'
	--exclude='.git/'
	--exclude='.idea/'
	--exclude='.vscode/'
	--exclude='.DS_Store'
	-e "ssh -p ${REMOTE_PORT}"
)

if [[ "${DRY_RUN}" -eq 1 ]]; then
	RSYNC_OPTS+=(--dry-run)
elif [[ "${DELETE_SYNC}" -eq 1 ]]; then
	RSYNC_OPTS+=(--delete)
fi

echo "[INFO] Source      : ${SOURCE_DIR}"
echo "[INFO] Destination : ${DESTINATION}"
echo "[INFO] Using ignore: ${SCRIPT_DIR}/.gitignore"
if [[ "${DRY_RUN}" -eq 1 ]]; then
	echo "[INFO] Mode        : dry-run (preview only)"
if [[ "${DELETE_SYNC}" -eq 1 ]]; then
	echo "[INFO] --delete ignored in dry-run mode; add --apply/-a to enable it"
fi
else
	echo "[INFO] Mode        : apply (real sync)"
if [[ "${DELETE_SYNC}" -eq 1 ]]; then
	echo "[INFO] Delete mode : enabled (--delete)"
fi
fi

if [[ "${DRY_RUN}" -eq 0 && "${DELETE_SYNC}" -eq 1 ]]; then
	echo "[WARN] You are about to run rsync with --delete."
	echo "[WARN] Files existing only on remote target may be removed."
	read -r -p "Type DELETE to continue: " confirm
	if [[ "${confirm}" != "DELETE" ]]; then
		echo "[INFO] Aborted by user."
		exit 1
	fi
fi

rsync "${RSYNC_OPTS[@]}" "${SOURCE_DIR}" "${DESTINATION}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[INFO] Dry-run complete. No changes were made."
else
    echo "[INFO] Sync completed."
fi

# ./rsync_h800.sh
# ./rsync_h800.sh --apply
# ./rsync_h800.sh --apply --delete # dangerous, use with caution after confirming the prompt