"""Backend interfaces for synthetic tissue simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from omegaconf import DictConfig

from bpc_fno.simulation.tissue.conductivity import ConductivityTensor
from bpc_fno.simulation.tissue.geometry import VentricularSlab


@dataclass(slots=True)
class SimulationContext:
    """Immutable per-sample context passed into a simulation backend."""

    config: DictConfig
    slab: VentricularSlab
    conductivity: ConductivityTensor
    sdf: np.ndarray
    fiber: np.ndarray
    cell_type_map: np.ndarray
    fibrosis_mask: np.ndarray
    pacing_site_voxel: tuple[int, int, int]
    params: Mapping[str, Any]
    output_times_ms: np.ndarray
    save_vm: bool


@dataclass(slots=True)
class SimulationResult:
    """Output expected from all simulation backends."""

    J_i: np.ndarray
    t_ms: np.ndarray
    activation_times_ms: np.ndarray
    stimulus_mask: np.ndarray
    V_m: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SimulationBackend(ABC):
    """Abstract interface for tissue-field simulators."""

    name: str = "unknown"
    smoke_only: bool = False
    requires_lookup_ionics: bool = False
    coupling_description: str = "unspecified"

    @abstractmethod
    def simulate(
        self,
        context: SimulationContext,
        ap_waveforms: dict[int, dict[str, np.ndarray]] | None = None,
    ) -> SimulationResult:
        """Run the backend and return regularly sampled voxel-grid outputs."""
