#!/bin/bash
# Shared helpers for ops scripts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
RUNS_DIR="${PROJECT_ROOT}/runs"

mkdir -p "${LOG_DIR}" "${RUNS_DIR}"

resolve_python() {
    if [ -x "${PROJECT_ROOT}/venv/bin/python3" ]; then
        echo "${PROJECT_ROOT}/venv/bin/python3"
        return 0
    fi
    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return 0
    fi
    echo "python3_not_found"
    return 1
}

PYTHON_BIN="$(resolve_python || true)"

require_python() {
    if [ -z "${PYTHON_BIN}" ] || [ "${PYTHON_BIN}" = "python3_not_found" ]; then
        echo "[ERROR] python3 not found."
        exit 127
    fi
}

ensure_project_root() {
    cd "${PROJECT_ROOT}"
}
