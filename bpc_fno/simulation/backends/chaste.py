"""Chaste backend placeholder.

The production path in this repository targets openCARP on Linux. Chaste is
kept as an architectural stub so the backend interface remains explicit and
extensible without pretending that a tested implementation exists here.
"""

from __future__ import annotations

from bpc_fno.simulation.backends.base import (
    SimulationBackend,
    SimulationContext,
    SimulationResult,
)


class ChasteBackend(SimulationBackend):
    """Documented but intentionally unimplemented backend."""

    name = "chaste"
    requires_lookup_ionics = False
    coupling_description = "not_implemented"

    def simulate(
        self,
        context: SimulationContext,
        ap_waveforms=None,
    ) -> SimulationResult:
        _ = (context, ap_waveforms)
        raise NotImplementedError(
            "ChasteBackend is a documented interface stub only. "
            "The production backend in this repository is openCARP on Linux."
        )
