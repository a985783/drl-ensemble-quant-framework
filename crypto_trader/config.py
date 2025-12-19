"""
Configuration Loader | 配置加载器

Loads configuration with priority:
1. config.yaml (local, gitignored) - Your live trading parameters
2. config_template.yaml (public defaults) - Textbook values

配置加载优先级：
1. config.yaml（本地，已 gitignore）- 您的实盘参数
2. config_template.yaml（公开默认值）- 教科书值

Usage | 使用方法:
    from config import CONFIG
    rsi_period = CONFIG['features']['rsi_period']
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config() -> Dict[str, Any]:
    """
    Load configuration with fallback to template.
    
    Returns:
        dict: Configuration dictionary
        
    Raises:
        FileNotFoundError: If no config file found
    """
    root = Path(__file__).parent.parent
    local_config = root / "config.yaml"
    template_config = root / "config_template.yaml"
    
    config_path = None
    
    if local_config.exists():
        config_path = local_config
        print(f"[Config] Loading local config: {local_config}")
    elif template_config.exists():
        config_path = template_config
        print(f"[Config] Loading template config: {template_config}")
    else:
        raise FileNotFoundError(
            "No configuration file found! "
            "Please create config.yaml or config_template.yaml in project root."
        )
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def get_nested(config: Dict, *keys, default=None):
    """
    Safely get nested config value.
    
    Example:
        get_nested(CONFIG, 'features', 'rsi_period', default=14)
    """
    result = config
    for key in keys:
        if isinstance(result, dict) and key in result:
            result = result[key]
        else:
            return default
    return result


# Global config instance - loaded once at import
CONFIG = load_config()


# Convenience accessors
def features_config() -> Dict:
    """Get features configuration section."""
    return CONFIG.get('features', {})


def training_config() -> Dict:
    """Get training configuration section."""
    return CONFIG.get('training', {})


def environment_config() -> Dict:
    """Get environment configuration section."""
    return CONFIG.get('environment', {})


def risk_config() -> Dict:
    """Get risk management configuration section."""
    return CONFIG.get('risk', {})


def signal_model_config() -> Dict:
    """Get signal model configuration section."""
    return CONFIG.get('signal_model', {})


def live_trading_config() -> Dict:
    """Get live trading configuration section."""
    return CONFIG.get('live_trading', {})
