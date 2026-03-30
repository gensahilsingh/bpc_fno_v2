"""Monodomain PDE residual loss for physics-informed training.

Unit system
-----------
- V_m : transmembrane potential in mV
- J_i : impressed (intracellular) current density in uA/cm^2
- Distances : cm
- Conductivities : mS/cm
- C_m : membrane capacitance in uF/cm^2
- beta : surface-to-volume ratio in cm^-1

Monodomain equation (reference)
-------------------------------
    div(D_i * grad(V_m)) = beta * C_m * dV_m/dt + beta * I_ion(V_m, w)

For Architecture A we work with J_i = -D_i * grad(V_m) directly, so:
    div(J_i) = -(beta * C_m * dV_m/dt + beta * I_ion)

In quasi-static regions (away from the activation front), dV_m/dt ~ 0
and I_ion ~ g_total*(V_m - V_rest) ~ 0 for resting tissue, giving:
    div(J_i) ~ 0   (current conservation)

This module penalises non-zero divergence of J_i inside the myocardium,
encouraging physically consistent current fields.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class MonodomainPDELoss(nn.Module):
    """Penalise violation of the monodomain current-conservation constraint.

    Computes ``R = div(J_i)`` via central finite differences and returns
    ``mean(R^2)`` evaluated at randomly sampled collocation points inside
    the myocardium mask.
    """

    # ------------------------------------------------------------------ #
    # Physical constants (simplified ionic closure)
    # ------------------------------------------------------------------ #
    BETA: float = 0.14              # cm^-1  surface-to-volume ratio
    C_M: float = 1.0               # uF/cm^2  membrane capacitance
    G_TOTAL: float = 0.025         # mS/cm^2  linearised total ionic conductance
    V_REST: float = -85.0          # mV  resting transmembrane potential

    def __init__(
        self,
        config: DictConfig,
        conductivity_inverse: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        self.voxel_size_cm: float = float(config.get("voxel_size_cm", 0.1))
        self.n_collocation_points: int = int(
            config.get("n_collocation_points", 1024)
        )

        # Optional inverse intracellular conductivity tensor D_i^{-1}.
        # Shape: (3, 3) or (B, 3, 3) — used to approximate V_m from J_i.
        if conductivity_inverse is not None:
            self.register_buffer(
                "D_i_inv", conductivity_inverse.detach().clone()
            )
        else:
            self.D_i_inv: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    # Divergence via central finite differences
    # ------------------------------------------------------------------ #
    def compute_divergence_J(
        self, J_i: torch.Tensor, voxel_size: float
    ) -> torch.Tensor:
        """Compute div(J_i) = dJ_x/dx + dJ_y/dy + dJ_z/dz.

        Uses second-order central finite differences with zero-padding at
        the boundaries.

        Args:
            J_i: Current density field, shape ``(B, 3, N, N, N)``
                 (channels-first: J_x, J_y, J_z along dim=1).
            voxel_size: Spatial step in cm.

        Returns:
            div_J: Divergence field, shape ``(B, 1, N, N, N)``.
        """
        inv_2h = 1.0 / (2.0 * voxel_size)

        # dJ_x / dx  — differences along spatial dim 2 (first spatial axis)
        dJx_dx = torch.zeros_like(J_i[:, 0:1])
        dJx_dx[:, :, 1:-1, :, :] = (
            J_i[:, 0:1, 2:, :, :] - J_i[:, 0:1, :-2, :, :]
        ) * inv_2h

        # dJ_y / dy  — differences along spatial dim 3
        dJy_dy = torch.zeros_like(J_i[:, 0:1])
        dJy_dy[:, :, :, 1:-1, :] = (
            J_i[:, 1:2, :, 2:, :] - J_i[:, 1:2, :, :-2, :]
        ) * inv_2h

        # dJ_z / dz  — differences along spatial dim 4
        dJz_dz = torch.zeros_like(J_i[:, 0:1])
        dJz_dz[:, :, :, :, 1:-1] = (
            J_i[:, 2:3, :, :, 2:] - J_i[:, 2:3, :, :, :-2]
        ) * inv_2h

        return dJx_dx + dJy_dy + dJz_dz  # (B, 1, N, N, N)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        J_i_hat: torch.Tensor,
        geometry: torch.Tensor,
        myocardium_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the monodomain PDE residual loss.

        The physically meaningful residual is ``div(J_i)``, which should
        be approximately zero in quasi-static resting tissue (current
        conservation).  We sample ``n_collocation_points`` inside the
        myocardium and return ``mean(R^2)``.

        Args:
            J_i_hat: Predicted current density, ``(B, 3, N, N, N)``.
            geometry: Geometry tensor, ``(B, 4, N, N, N)``
                      (SDF, mu_r, sigma, mask).  The mask channel (index 3)
                      is used as the myocardium mask when *myocardium_mask*
                      is not provided.
            myocardium_mask: Optional boolean / float mask ``(B, 1, N, N, N)``.
                             If ``None``, the mask channel of *geometry* is
                             used.

        Returns:
            Scalar loss tensor (mean squared residual at collocation points).
        """
        B, _, N, _, _ = J_i_hat.shape

        # -- 1. Divergence residual ------------------------------------ #
        div_J = self.compute_divergence_J(J_i_hat, self.voxel_size_cm)
        # div_J: (B, 1, N, N, N)

        # -- 2. (Optional) approximate V_m correction ------------------ #
        # When D_i_inv is available we can estimate:
        #   grad(V_m) = -D_i_inv @ J_i  =>  V_m via cumulative sum
        # and add a simplified ionic current term.
        # For Architecture A we rely on the quasi-static approximation
        # and set R = div(J_i) directly.
        # TODO: Architecture C will incorporate the full TT2006 ionic
        #       model for the I_ion term here.
        residual = div_J  # (B, 1, N, N, N)

        # -- 3. Determine the mask ------------------------------------- #
        if myocardium_mask is None:
            # Use the mask channel (index 3) from the geometry tensor.
            myocardium_mask = geometry[:, 3:4, :, :, :]  # (B, 1, N, N, N)
        myocardium_mask = myocardium_mask.bool()

        # -- 4. Sample collocation points ------------------------------ #
        # Flatten the residual and mask across all batches for sampling.
        R_flat = residual.reshape(B, -1)          # (B, M)  where M = N^3
        mask_flat = myocardium_mask.reshape(B, -1)  # (B, M)

        loss = torch.tensor(0.0, device=J_i_hat.device, dtype=J_i_hat.dtype)

        for b in range(B):
            valid_idx = mask_flat[b].nonzero(as_tuple=False).squeeze(-1)
            if valid_idx.numel() == 0:
                continue

            n_pts = min(self.n_collocation_points, valid_idx.numel())
            perm = torch.randperm(
                valid_idx.numel(), device=J_i_hat.device
            )[:n_pts]
            sampled_idx = valid_idx[perm]

            R_sampled = R_flat[b, sampled_idx]
            loss = loss + (R_sampled ** 2).mean()

        loss = loss / max(B, 1)
        return loss
