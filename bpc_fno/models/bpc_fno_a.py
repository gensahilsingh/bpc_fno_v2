"""Top-level Architecture A: BPC-FNO with shared FNO backbone."""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from bpc_fno.models.fno_layers import FNOBackbone
from bpc_fno.models.forward_pino import ForwardPINO
from bpc_fno.models.geometry_encoder import VoxelGeometryEncoder
from bpc_fno.models.inverse_encoder import InverseEncoder
from bpc_fno.models.vae_decoder import VAEDecoder


class BPC_FNO_A(nn.Module):
    """BPC-FNO Architecture A.

    Shared-weight architecture where a single :class:`VoxelGeometryEncoder`
    and a single :class:`FNOBackbone` are referenced by both the forward
    operator (:class:`ForwardPINO`) and the inverse encoder
    (:class:`InverseEncoder`).  The VAE decoder is independent.

    Parameters
    ----------
    config:
        Hydra/OmegaConf configuration with at least ``config.model`` containing:
        - ``n_fno_hidden`` (int): hidden channel width (e.g. 64)
        - ``n_fno_layers`` (int): number of FNO spectral blocks
        - ``n_fno_modes`` (int): Fourier modes per axis
        - ``n_geometry_channels`` (int, default 4)
        - ``latent_dim`` (int, default 512)
        - ``n_sensors_total`` (int, default 48)
        - ``n_output_timesteps`` (int, default 50)
        - ``grid_size`` (int, default 32)
        - ``decoder_base_channels`` (int, default 512)
        - ``n_decoder_upsamples`` (int, default 3)
        - ``use_bilinear_decoder`` (bool, default False)
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        m = config.model
        c_hidden: int = m.n_fno_hidden
        n_geo_ch: int = m.get("n_geometry_channels", 4)

        # ----- shared modules (instantiated ONCE) -----
        self.geometry_encoder = VoxelGeometryEncoder(n_geo_ch, c_hidden)
        self.fno_backbone = FNOBackbone(
            c_in=c_hidden,
            c_out=c_hidden,
            n_layers=m.n_fno_layers,
            n_modes=m.n_fno_modes,
            c_hidden=c_hidden,
        )

        # ----- stage-specific modules (reference shared instances) -----
        self.forward_pino = ForwardPINO(
            self.geometry_encoder, self.fno_backbone, config
        )
        self.inverse_encoder = InverseEncoder(
            self.geometry_encoder, self.fno_backbone, config
        )
        self.vae_decoder = VAEDecoder(config)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_B_obs(batch: dict[str, Tensor]) -> Tensor:
        """Resolve the observed B-field from either the new or legacy batch key."""
        for key in ("B_obs", "B_mig", "B_true"):
            if key in batch:
                return batch[key]
        raise KeyError("Batch must contain one of: 'B_obs', 'B_mig', or 'B_true'.")

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Full forward pass through all stages.

        Args:
            batch: dict with keys
                - ``J_i``: (B, 3, N, N, N) or (B, 3, T, N, N, N)
                - ``geometry``: (B, 4, N, N, N)
                - ``B_obs`` or legacy ``B_mig``: (B, N_sensors*3[, T])

        Returns:
            dict with keys ``B_pred``, ``J_i_hat``, ``mu``, ``log_var``, ``z``.
        """
        J_i = batch["J_i"]
        geometry = batch["geometry"]
        B_obs = self._resolve_B_obs(batch)

        # Stage 2: forward operator
        B_pred = self.forward_pino.predict_B(J_i, geometry)

        # Stage 3: inverse encoder
        mu, log_var = self.inverse_encoder.encode_to_latent(B_obs, geometry)
        z = self.inverse_encoder.sample_z(mu, log_var)

        # Stage 4: decoder
        J_i_hat = self.vae_decoder.decode(z, geometry)

        return {
            "B_pred": B_pred,
            "J_i_hat": J_i_hat,
            "mu": mu,
            "log_var": log_var,
            "z": z,
        }

    def forward_only(self, J_i: Tensor, geometry: Tensor) -> Tensor:
        """Phase 1 training: forward path only.

        Args:
            J_i: (B, 3, N, N, N) or (B, 3, T, N, N, N)
            geometry: (B, 4, N, N, N)

        Returns:
            B_pred: (B, N_sensors*3) or (B, N_sensors*3, T)
        """
        return self.forward_pino.predict_B(J_i, geometry)

    @torch.no_grad()
    def reconstruct(
        self,
        B_obs: Tensor,
        geometry: Tensor,
        n_samples: int = 1,
    ) -> dict[str, Tensor]:
        """Inference: draw multiple posterior samples for uncertainty
        quantification.

        Args:
            B_obs: (B, N_sensors*3, T)
            geometry: (B, 4, N, N, N)
            n_samples: number of latent samples to draw.

        Returns:
            dict with keys:
            - ``J_i_mean``: decoder-shaped posterior mean
            - ``J_i_std``:  decoder-shaped posterior std
            - ``mu``:       (B, D)
            - ``log_var``:  (B, D)
        """
        was_training = self.training
        self.eval()

        mu, log_var = self.inverse_encoder.encode_to_latent(B_obs, geometry)

        samples: list[Tensor] = []
        std = (0.5 * log_var).exp()
        for _ in range(n_samples):
            eps = torch.randn_like(std)
            z = mu + eps * std
            J_i_hat = self.vae_decoder.decode(z, geometry)
            samples.append(J_i_hat)

        stacked = torch.stack(samples, dim=0)
        J_i_mean = stacked.mean(dim=0)
        J_i_std = stacked.std(dim=0) if n_samples > 1 else torch.zeros_like(J_i_mean)

        if was_training:
            self.train()

        return {
            "J_i_mean": J_i_mean,
            "J_i_std": J_i_std,
            "mu": mu,
            "log_var": log_var,
        }

    # ------------------------------------------------------------------
    # Parameter groups for differential learning rates
    # ------------------------------------------------------------------

    def get_parameter_groups(
        self, lr: float | None = None
    ) -> dict[str, list[nn.Parameter]]:
        """Return named parameter groups for the optimiser.

        Groups
        ------
        - ``fno_shared``: geometry encoder + FNO backbone
        - ``forward_head``: forward-PINO-only params (input adapter, sensor head)
        - ``inverse_head``: inverse-encoder-only params (B projection, adapter, VAE heads)
        - ``decoder``: VAE decoder params

        Returns:
            Mapping from group name to list of parameters.
        """
        shared_ids = {
            id(p)
            for p in list(self.geometry_encoder.parameters())
            + list(self.fno_backbone.parameters())
        }

        forward_head_params = [
            p
            for p in self.forward_pino.parameters()
            if id(p) not in shared_ids
        ]
        inverse_head_params = [
            p
            for p in self.inverse_encoder.parameters()
            if id(p) not in shared_ids
        ]

        return {
            "fno_shared": list(self.geometry_encoder.parameters())
            + list(self.fno_backbone.parameters()),
            "forward_head": forward_head_params,
            "inverse_head": inverse_head_params,
            "decoder": list(self.vae_decoder.parameters()),
        }
