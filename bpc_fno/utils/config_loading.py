"""Helpers for loading OmegaConf configs with lightweight inheritance."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config_with_extends(path: str | Path) -> DictConfig:
    """Load a YAML config supporting a simple ``extends`` key."""
    path = Path(path)
    cfg = OmegaConf.load(path)

    extends = cfg.get("extends")
    if not extends:
        return cfg

    parent_path = Path(extends)
    if not parent_path.is_absolute():
        parent_path = (path.parent / parent_path).resolve()

    cfg = OmegaConf.to_container(cfg, resolve=False)
    cfg.pop("extends", None)
    child_cfg = OmegaConf.create(cfg)
    parent_cfg = load_config_with_extends(parent_path)
    return OmegaConf.merge(parent_cfg, child_cfg)
