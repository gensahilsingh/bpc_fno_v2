"""Simulation backend registry."""

from __future__ import annotations

from bpc_fno.simulation.backends.chaste import ChasteBackend
from bpc_fno.simulation.backends.eikonal import EikonalBackend
from bpc_fno.simulation.backends.opencarp import OpenCARPBackend
from bpc_fno.simulation.backends.windows_hybrid import (
    WindowsHybridMonodomainBackend,
)

BACKEND_REGISTRY = {
    "eikonal": EikonalBackend,
    "windows_hybrid": WindowsHybridMonodomainBackend,
    "opencarp": OpenCARPBackend,
    "chaste": ChasteBackend,
}
