#!/bin/bash
# Launch Agent wrapper script for daily trading scheduler
# Called by macOS launchd on login / system boot

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

ensure_project_root
require_python

LOG_FILE="${LOG_DIR}/daemon.log"

mkdir -p "${LOG_DIR}"

# Load environment
export CONFIRM_REAL_MONEY=True
export DOTENV_PATH=.env.live
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

# Logging function
log() {
    echo "[$(date)] $*" >> "${LOG_FILE}"
}

log "Launch Agent starting trading daemon..."
log "User: $(whoami)"
log "Path: $PATH"
log "Using python: ${PYTHON_BIN}"

# DEBUG: Check if python is actually executable
if [ ! -x "${PYTHON_BIN}" ]; then
    log "Error: Python interpreter not executable: ${PYTHON_BIN}"
    ls -l "${PYTHON_BIN}" >> "${LOG_FILE}" 2>&1
    exit 126
fi

# Resolve caffeinate path
CAFFEINATE="/usr/bin/caffeinate"
if [ ! -x "$CAFFEINATE" ]; then
    log "Error: caffeinate not found at $CAFFEINATE"
    exit 127
fi

log "Executing: $CAFFEINATE -d -i -m -s ${PYTHON_BIN} -u crypto_trader/start_simulation.py"

exec "$CAFFEINATE" -d -i -m -s "${PYTHON_BIN}" -u "crypto_trader/start_simulation.py" >> "${LOG_FILE}" 2>&1
