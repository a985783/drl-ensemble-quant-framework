#!/bin/bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
ensure_project_root
require_python

export DOTENV_PATH="${DOTENV_PATH:-.env.live}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/monitor_strategy_effectiveness.py" --mode weekly --send-alert "$@"
