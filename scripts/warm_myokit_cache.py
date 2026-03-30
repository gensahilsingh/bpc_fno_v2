"""Pre-warm Myokit compilation cache for all 3 TT2006 cell types.

Ensures the compiled .pyd simulation files are cached before bulk
data generation, avoiding the WinError 206 os.add_dll_directory bug
that occurs during fresh compilation on Windows with long paths.
"""

from __future__ import annotations

import os
import tempfile

# Redirect temp to short path BEFORE any myokit import
_MYOKIT_TMP = os.path.join("C:\\", "tmp", "mk")
os.makedirs(_MYOKIT_TMP, exist_ok=True)
os.environ["TEMP"] = _MYOKIT_TMP
os.environ["TMP"] = _MYOKIT_TMP
os.environ["TMPDIR"] = _MYOKIT_TMP
tempfile.tempdir = _MYOKIT_TMP

import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    import myokit
    from omegaconf import OmegaConf

    from bpc_fno.simulation.ionic.cellml_loader import CellMLLoader

    cfg = OmegaConf.create({
        "simulation": {
            "ionic": OmegaConf.load("configs/data_gen.yaml").ionic
        }
    })
    loader = CellMLLoader(cfg)

    for cell_type in ("endo", "mid", "epi"):
        logger.info("Warming cache for %s...", cell_type)
        model = loader.get_model(cell_type)
        model = model.clone()

        # Bind stimulus to pace protocol
        stim_name = None
        for var in model.variables(deep=True):
            if var.qname().endswith("i_Stim"):
                stim_name = var.qname()
                break
        if stim_name:
            model.get(stim_name).set_binding("pace")

        protocol = myokit.pacing.blocktrain(
            period=1000, duration=2.0, offset=10, level=-52
        )

        try:
            sim = myokit.Simulation(model, protocol)
            sim.run(1.0)
            logger.info("  %s: OK (compilation and 1ms run succeeded)", cell_type)
        except Exception as exc:
            logger.error("  %s: FAILED — %s", cell_type, exc)
            sys.exit(1)

    logger.info("Cache warm for: endo, mid, epi")


if __name__ == "__main__":
    main()
