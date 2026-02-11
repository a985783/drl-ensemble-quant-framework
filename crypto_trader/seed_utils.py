"""
seed_utils.py - Fixed seed management for reproducibility

Provides:
- set_global_seeds(seed): Sets all random seeds (numpy, random, torch, SB3)
- get_run_id(): Generates unique run identifier (timestamp + git hash)
"""
import os
import random
import subprocess
from datetime import datetime, timezone
from typing import Optional

import numpy as np


def set_global_seeds(seed: int) -> None:
    """
    Set all random seeds for reproducibility.
    
    Covers:
    - Python random
    - NumPy
    - PyTorch (if available)
    - Stable-Baselines3 (via environment variable)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Set environment variable for SB3 reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # PyTorch (optional dependency)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # PyTorch not installed


def get_git_short_hash() -> str:
    """Get short git commit hash, or 'nogit' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return "nogit"


def get_run_id(prefix: Optional[str] = None) -> str:
    """
    Generate unique run identifier.
    
    Format: YYYYMMDD_HHMMSS_<git_short_hash>
    Example: 20251226_130544_abc1234
    
    Args:
        prefix: Optional prefix for the run_id (e.g., "backtest", "train")
    
    Returns:
        Unique run identifier string
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    git_hash = get_git_short_hash()
    
    if prefix:
        return f"{prefix}_{timestamp}_{git_hash}"
    return f"{timestamp}_{git_hash}"
