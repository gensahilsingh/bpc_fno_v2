from __future__ import annotations

from pathlib import Path

from bpc_fno.utils.config_loading import load_config_with_extends


def test_load_config_with_extends(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"

    base.write_text(
        "simulation:\n"
        "  pipeline: eikonal\n"
        "  n_samples: 10\n"
        "monodomain:\n"
        "  dt_ms: 0.05\n",
        encoding="utf-8",
    )
    child.write_text(
        "extends: base.yaml\n"
        "simulation:\n"
        "  n_samples: 5\n",
        encoding="utf-8",
    )

    cfg = load_config_with_extends(child)

    assert cfg.simulation.pipeline == "eikonal"
    assert int(cfg.simulation.n_samples) == 5
    assert float(cfg.monodomain.dt_ms) == 0.05
