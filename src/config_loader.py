"""Centralised configuration loader.

BUG-1 fix: track the loaded path so that calling ``load_config`` with a
different path either reloads or warns, rather than silently returning
stale data.
"""

import logging
from typing import Any, Dict, Optional

import yaml

log = logging.getLogger(__name__)

_CONFIG: Optional[Dict[str, Any]] = None
_CONFIG_PATH: Optional[str] = None


def load_config(path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load and cache YAML configuration.

    If the config has already been loaded from a *different* ``path``, a
    warning is logged and the config is reloaded from the new path so that
    the caller always gets the configuration they asked for.
    """
    global _CONFIG, _CONFIG_PATH

    if _CONFIG is not None:
        if _CONFIG_PATH == path:
            return _CONFIG
        log.warning(
            "load_config called with path '%s' but config was already loaded "
            "from '%s'. Reloading from the new path.",
            path,
            _CONFIG_PATH,
        )

    with open(path, "r", encoding="utf-8") as f:
        _CONFIG = yaml.safe_load(f)
    _CONFIG_PATH = path
    return _CONFIG


def reload_config(path: str = "config/config.yaml") -> Dict[str, Any]:
    """Force-reload configuration from disk."""
    global _CONFIG, _CONFIG_PATH
    _CONFIG = None
    _CONFIG_PATH = None
    return load_config(path)
