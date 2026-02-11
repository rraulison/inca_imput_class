"""Carregador centralizado de configuracao (singleton)."""

import yaml

_CONFIG = None


def load_config(path="config/config.yaml"):
    global _CONFIG
    if _CONFIG is None:
        with open(path, "r", encoding="utf-8") as f:
            _CONFIG = yaml.safe_load(f)
    return _CONFIG


def reload_config(path="config/config.yaml"):
    global _CONFIG
    _CONFIG = None
    return load_config(path)
