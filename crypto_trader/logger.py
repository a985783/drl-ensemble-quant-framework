"""
logger.py - Unified logging with run_id archiving

Provides:
- setup_logging(run_id): Configures logging with file + console output
- get_logger(name): Returns a logger instance
"""
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Global run directory (set by setup_logging)
_RUN_DIR: Optional[Path] = None


def get_runs_base_dir() -> Path:
    """Get the base directory for all runs."""
    # Project root is parent of crypto_trader/
    project_root = Path(__file__).parent.parent
    return project_root / "runs"


def setup_logging(
    run_id: str,
    level: int = logging.INFO,
    console: bool = True
) -> Path:
    """
    Configure logging for a run.
    
    Creates directory: runs/<run_id>/
    Writes logs to: runs/<run_id>/run.log
    
    Args:
        run_id: Unique run identifier
        level: Logging level (default: INFO)
        console: Whether to also log to console (default: True)
    
    Returns:
        Path to the run directory
    """
    global _RUN_DIR
    
    # Create run directory
    run_dir = get_runs_base_dir() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _RUN_DIR = run_dir
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # File handler
    log_file = run_dir / "run.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # Log startup info
    root_logger.info(f"Run ID: {run_id}")
    root_logger.info(f"Log directory: {run_dir}")
    
    return run_dir


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__ of the module)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def get_run_dir() -> Optional[Path]:
    """
    Get the current run directory.
    
    Returns:
        Path to run directory, or None if setup_logging() hasn't been called
    """
    return _RUN_DIR
