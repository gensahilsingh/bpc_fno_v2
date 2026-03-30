"""Bidomain PDE residual loss — stub for Architecture B/C upgrade.

Bidomain equations
------------------
The full bidomain model couples intracellular and extracellular potentials:

    div(D_i * grad(V_m)) + div(D_i * grad(phi_e))
        = beta * C_m * dV_m/dt + beta * I_ion(V_m, w)

    div((D_i + D_e) * grad(phi_e)) + div(D_i * grad(V_m)) = 0

where:
    V_m    — transmembrane potential (mV)
    phi_e  — extracellular potential (mV)
    D_i    — intracellular conductivity tensor (mS/cm)
    D_e    — extracellular conductivity tensor (mS/cm)
    beta   — surface-to-volume ratio (cm^-1)
    C_m    — membrane capacitance (uF/cm^2)
    I_ion  — total ionic current density from cell model (uA/cm^2)
    w      — vector of ionic gating variables

Architecture B will solve the bidomain forward problem with a learned
operator; Architecture C will couple the full TT2006 ionic model to
provide I_ion.

This module is intentionally left as a stub.  It will be implemented
when the project advances beyond Architecture A (monodomain / quasi-static).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BidomainPDELoss(nn.Module):
    """Bidomain PDE residual loss (not yet implemented).

    Raises :class:`NotImplementedError` on construction and on forward.
    Reserved for Architecture B/C upgrade path.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(
            "BidomainPDELoss is reserved for Architecture B/C.  "
            "The bidomain equations require coupled intracellular / "
            "extracellular conductivity tensors (D_i, D_e) and a full "
            "ionic cell model (e.g. TT2006) which are not yet integrated.  "
            "Use MonodomainPDELoss for Architecture A."
        )

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise NotImplementedError(
            "BidomainPDELoss.forward() is not implemented.  "
            "See class docstring for details on the Architecture B/C "
            "upgrade path."
        )
