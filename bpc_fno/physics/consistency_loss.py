"""Forward-consistency loss for physics-informed inverse training.

Ensures that the predicted current density J_i_hat, when passed through
the (differentiable) forward operator, reproduces the observed magnetic
field measurements B_obs.

    L_consistency = MSE( forward_model(J_i_hat, geometry),  B_obs )

Gradients flow through the forward model into J_i_hat (and therefore
into the decoder that produced it).  Whether the forward model's own
weights are updated by this loss is controlled externally — typically
the forward model parameters are frozen or placed in a separate
optimizer group with zero learning rate.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bpc_fno.models.interfaces import ForwardOperatorInterface


class ForwardConsistencyLoss(nn.Module):
    """MSE between re-predicted B-field and observed B-field."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        forward_model: ForwardOperatorInterface,
        J_i_hat: torch.Tensor,
        geometry: torch.Tensor,
        B_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute forward-consistency loss.

        Args:
            forward_model: A differentiable forward operator implementing
                :meth:`predict_B`.  Gradients propagate through this
                operator to *J_i_hat*, but the caller is responsible for
                ensuring the forward model's weights are not updated
                (e.g. by excluding them from the optimizer or freezing
                parameters).
            J_i_hat: Predicted intracellular current density,
                     shape ``(B, 3, N, N, N)``.
            geometry: Voxelised geometry channels,
                      shape ``(B, 4, N, N, N)``.
            B_obs: Observed magnetic field measurements,
                   shape ``(B, N_sensors*3, T)``.

        Returns:
            Scalar MSE loss tensor.
        """
        B_check: torch.Tensor = forward_model.predict_B(J_i_hat, geometry)
        return F.mse_loss(B_check, B_obs)
