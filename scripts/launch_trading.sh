#!/bin/bash
# Launch Agent wrapper script for daily trading scheduler
# Called by macOS launchd on login / system boot

PROJECT_ROOT="/Users/cuiqingsong/Documents/强化学习 i"
LOG_FILE="${PROJECT_ROOT}/runs/daemon.log"

cd "${PROJECT_ROOT}" || (echo "Failed to cd to ${PROJECT_ROOT}" >> "${LOG_FILE}"; exit 1)

# Ensure log directory
mkdir -p "${PROJECT_ROOT}/runs"

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

# Activate venv
if [ -f "${PROJECT_ROOT}/venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/venv/bin/activate"
else
    log "Warning: venv activate not found"
fi

# Determine python
if [ -x "${PROJECT_ROOT}/venv/bin/python3" ]; then
    PYTHON="${PROJECT_ROOT}/venv/bin/python3"
    log "Using venv python: $PYTHON"
else
    PYTHON="/usr/bin/python3"
    log "Using system python: $PYTHON"
fi

# DEBUG: Check if python is actually executable
if [ ! -x "$PYTHON" ]; then
    log "Error: Python interpreter not executable: $PYTHON"
    ls -l "$PYTHON" >> "${LOG_FILE}" 2>&1
    exit 126
fi

# Use caffeinate to prevent sleep; redirect output to log files
CMD="/usr/local/bin/python3"
# Resolve caffeinate path
CAFFEINATE="/usr/bin/caffeinate"
if [ ! -x "$CAFFEINATE" ]; then
    log "Error: caffeinate not found at $CAFFEINATE"
    exit 127
fi

log "Executing: $CAFFEINATE -d -i -m -s $PYTHON -u crypto_trader/start_simulation.py"

exec "$CAFFEINATE" -d -i -m -s "$PYTHON" -u "crypto_trader/start_simulation.py" >> "${LOG_FILE}" 2>&1
