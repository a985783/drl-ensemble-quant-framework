#!/usr/bin/env python3
"""
OKX 实盘交易启动脚本
默认使用 .env.live 配置

⚠️ 警告：此脚本使用真实资金交易！
"""
import os
import sys
from typing import Dict, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

REQUIRED_KEYS = ("OKX_API_KEY", "OKX_SECRET_KEY", "OKX_PASSPHRASE")
PLACEHOLDER_MARKERS = (
    "YOUR_",
    "REPLACE_",
    "CHANGE_ME",
    "PLACEHOLDER",
    "<YOUR",
    "PASTE_",
    "INPUT_",
    "EXAMPLE",
)


def _parse_env_file(path: str) -> Dict[str, str]:
    env_map: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            env_map[key.strip()] = value.strip().strip("'\"")
    return env_map


def _looks_placeholder(value: str) -> bool:
    val = value.strip()
    if not val:
        return True
    upper = val.upper()
    if any(marker in upper for marker in PLACEHOLDER_MARKERS):
        return True
    if len(val) >= 4 and set(val) <= {"x", "X", "*", "_", "-"}:
        return True
    return False


def _validate_live_config(env_map: Dict[str, str]) -> List[str]:
    errors: List[str] = []
    for key in REQUIRED_KEYS:
        value = env_map.get(key, "")
        if _looks_placeholder(value):
            errors.append(f"{key} 未配置有效值")

    demo_mode = env_map.get("OKX_DEMO_MODE", "True").strip().lower()
    if demo_mode == "true":
        errors.append("OKX_DEMO_MODE=True（当前为模拟盘，不是实盘）")

    confirm_real_money = env_map.get("CONFIRM_REAL_MONEY", "False").strip().lower()
    if confirm_real_money != "true":
        errors.append("CONFIRM_REAL_MONEY 不是 True")
    return errors


def _prepare_live_env() -> str:
    live_env = os.path.join(BASE_DIR, ".env.live")
    if not os.path.exists(live_env):
        raise RuntimeError("找不到 .env.live 配置文件")

    env_map = _parse_env_file(live_env)
    validation_errors = _validate_live_config(env_map)
    if validation_errors:
        detail = "; ".join(validation_errors)
        raise RuntimeError(f".env.live 校验失败: {detail}")

    return live_env


def main() -> None:
    os.chdir(BASE_DIR)
    live_env = _prepare_live_env()
    os.environ["DOTENV_PATH"] = live_env

    print("⚠️  已加载实盘配置 (.env.live)")
    print("⚠️  警告：将使用真实资金交易！")
    print("")

    sys.path.insert(0, SCRIPT_DIR)
    from run_live import main as run_live_main

    run_live_main()


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"❌ {exc}")
        print(f"   位置: {BASE_DIR}/.env.live")
        raise SystemExit(1)
