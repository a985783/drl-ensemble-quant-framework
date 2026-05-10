import os
from typing import Any, Dict


def resolve_ccxt_proxies() -> Dict[str, str]:
    """Return requests-compatible proxy config for ccxt."""
    http_proxy = (
        os.getenv("OKX_HTTP_PROXY")
        or os.getenv("HTTP_PROXY")
        or os.getenv("http_proxy")
    )
    https_proxy = (
        os.getenv("OKX_HTTPS_PROXY")
        or os.getenv("HTTPS_PROXY")
        or os.getenv("https_proxy")
    )

    proxies: Dict[str, str] = {}
    if http_proxy:
        proxies["http"] = http_proxy
    if https_proxy:
        proxies["https"] = https_proxy
    return proxies


def apply_ccxt_proxy_config(exchange_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(exchange_cfg)
    proxies = resolve_ccxt_proxies()
    if proxies:
        cfg["proxies"] = proxies
    return cfg
