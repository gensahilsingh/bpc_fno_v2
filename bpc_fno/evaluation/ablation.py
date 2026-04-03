"""Stage-by-stage ablation evaluation harness for BPC-FNO.

Provides systematic ablation experiments to quantify the contribution of
each architectural component: physics loss, consistency loss, shared
weights, and uncertainty-quantification sample count.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from bpc_fno.evaluation.metrics import (
    MetricsComputer,
    _get_forward_target,
    _get_observed_B,
    pearson_correlation,
    relative_l2_error,
    uq_coverage,
    uq_sharpness,
)
from bpc_fno.models.bpc_fno_a import BPC_FNO_A

logger = logging.getLogger(__name__)


class AblationEvaluator:
    """Run ablation experiments on a trained BPC-FNO-A model.

    Parameters
    ----------
    model:
        Fully-trained BPC-FNO-A model (all stages).
    dataloader:
        Evaluation-split data loader.
    config:
        Hydra/OmegaConf run configuration.
    metrics_computer:
        Pre-configured :class:`MetricsComputer` instance.
    """

    def __init__(
        self,
        model: BPC_FNO_A,
        dataloader: DataLoader[Any],
        config: DictConfig,
        metrics_computer: MetricsComputer,
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.metrics = metrics_computer
        self.device = next(model.parameters()).device

    # ------------------------------------------------------------------
    # Helper: collect metrics over the full evaluation set
    # ------------------------------------------------------------------

    def _evaluate_loop(
        self,
        model: BPC_FNO_A,
    ) -> dict[str, float]:
        """Run ``model`` over the dataloader and aggregate metrics.

        Returns:
            Dict of mean metric values.
        """
        model.eval()
        accum: dict[str, list[float]] = {}
        n_batches = 0

        with torch.no_grad():
            for batch in self.dataloader:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                output = model(batch)
                batch_metrics = self.metrics.compute_all(output, batch, model)

                for key, val in batch_metrics.items():
                    accum.setdefault(key, []).append(val)
                n_batches += 1

        if n_batches == 0:
            logger.warning("Evaluation dataloader yielded zero batches.")
            return {}

        return {k: sum(v) / len(v) for k, v in accum.items()}

    # ------------------------------------------------------------------
    # Ablation: forward model only (Stage 2 isolation)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_forward_only(self) -> dict[str, float]:
        """Evaluate the forward operator (Stage 2) in isolation.

        Metrics computed: ``rel_l2_B``, ``pearson_r_B``.
        """
        self.model.eval()
        rel_l2_list: list[float] = []
        pearson_list: list[float] = []

        for batch in self.dataloader:
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            J_i = batch["J_i"]
            geometry = batch["geometry"]
            B_true = _get_forward_target(batch)

            B_pred = self.model.forward_only(J_i, geometry)

            rel_l2_list.append(
                float(relative_l2_error(B_pred, B_true).mean().item())
            )
            # Pearson R on (B, C, T) — treat sensor channels as the C dim
            # Reshape to (B, C, 1, 1, T) for the generic function then squeeze
            B_p = B_pred.unsqueeze(-2).unsqueeze(-2)  # (B, S, 1, 1, T)
            B_t = B_true.unsqueeze(-2).unsqueeze(-2)
            # pearson_correlation expects (B, C, N, N, N); we set N=1,1,T
            pr = pearson_correlation(B_p, B_t)  # (B, S)
            pearson_list.append(float(pr.mean().item()))

        results = {
            "forward_only/rel_l2_B": (
                sum(rel_l2_list) / len(rel_l2_list) if rel_l2_list else float("nan")
            ),
            "forward_only/pearson_r_B": (
                sum(pearson_list) / len(pearson_list) if pearson_list else float("nan")
            ),
        }
        logger.info("Forward-only evaluation: %s", results)
        return results

    # ------------------------------------------------------------------
    # Ablation: without physics loss
    # ------------------------------------------------------------------

    def evaluate_without_physics_loss(self) -> dict[str, float]:
        """Evaluate a model variant trained *without* the physics loss.

        Expects a checkpoint path at ``config.ablation.no_physics_ckpt``.
        Falls back to the current model with a warning if the path is
        not configured.
        """
        ckpt_path: str | None = OmegaConf.select(
            self.config, "ablation.no_physics_ckpt", default=None
        )
        if ckpt_path is None or not Path(ckpt_path).exists():
            logger.warning(
                "No checkpoint for 'no physics loss' ablation found at %s. "
                "Evaluating current model as a placeholder.",
                ckpt_path,
            )
            model = self.model
        else:
            model = BPC_FNO_A(self.config)
            state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state["model"] if "model" in state else state)
            model = model.to(self.device)

        results = self._evaluate_loop(model)
        return {f"no_physics/{k}": v for k, v in results.items()}

    # ------------------------------------------------------------------
    # Ablation: without consistency loss
    # ------------------------------------------------------------------

    def evaluate_without_consistency_loss(self) -> dict[str, float]:
        """Evaluate a model variant trained *without* the forward-inverse
        consistency loss.

        Expects ``config.ablation.no_consistency_ckpt``.
        """
        ckpt_path: str | None = OmegaConf.select(
            self.config, "ablation.no_consistency_ckpt", default=None
        )
        if ckpt_path is None or not Path(ckpt_path).exists():
            logger.warning(
                "No checkpoint for 'no consistency loss' ablation found at %s. "
                "Evaluating current model as a placeholder.",
                ckpt_path,
            )
            model = self.model
        else:
            model = BPC_FNO_A(self.config)
            state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state["model"] if "model" in state else state)
            model = model.to(self.device)

        results = self._evaluate_loop(model)
        return {f"no_consistency/{k}": v for k, v in results.items()}

    # ------------------------------------------------------------------
    # Ablation: without shared weights
    # ------------------------------------------------------------------

    def evaluate_without_shared_weights(self) -> dict[str, float]:
        """Evaluate a model variant with non-shared FNO backbones.

        .. note::
           This requires a separate model class with independent forward
           and inverse FNO backbones.  Currently stubbed — contributions
           welcome.

        Returns:
            Empty dict with a logged TODO.
        """
        # TODO: Implement non-shared-weight BPC_FNO_A variant.
        #       This would duplicate the FNOBackbone and
        #       VoxelGeometryEncoder so that forward_pino and
        #       inverse_encoder each own independent copies.
        logger.warning(
            "evaluate_without_shared_weights is not yet implemented. "
            "A non-shared-weight model class is required."
        )
        return {"no_shared_weights/status": float("nan")}

    # ------------------------------------------------------------------
    # Ablation: UQ quality vs number of posterior samples
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_uq_quality(
        self,
        n_samples_list: list[int] | None = None,
    ) -> dict[str, Any]:
        """Measure UQ metrics as a function of the number of posterior samples.

        Args:
            n_samples_list: List of sample counts to sweep. Defaults to
                ``[10, 25, 50, 100]``.

        Returns:
            Nested dict keyed by ``"uq/<n_samples>/coverage"`` and
            ``"uq/<n_samples>/sharpness"``.
        """
        if n_samples_list is None:
            n_samples_list = [10, 25, 50, 100]

        self.model.eval()
        results: dict[str, float] = {}

        for n_s in n_samples_list:
            coverage_vals: list[float] = []
            sharpness_vals: list[float] = []

            for batch in self.dataloader:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                J_i_true = batch["J_i"]
                geometry = batch["geometry"]
                B_obs = _get_observed_B(batch)

                # Draw posterior samples
                mu, log_var = self.model.inverse_encoder.encode_to_latent(
                    B_obs, geometry
                )
                std = (0.5 * log_var).exp()

                samples: list[torch.Tensor] = []
                for _ in range(n_s):
                    eps = torch.randn_like(std)
                    z = mu + eps * std
                    J_hat = self.model.vae_decoder.decode(z, geometry)
                    samples.append(J_hat)

                J_i_samples = torch.stack(samples, dim=0)  # (S, B, 3, N, N, N)

                cov = uq_coverage(J_i_samples, J_i_true, alpha=0.95)
                sharp = float(uq_sharpness(J_i_samples).item())

                coverage_vals.append(cov)
                sharpness_vals.append(sharp)

            mean_cov = sum(coverage_vals) / len(coverage_vals) if coverage_vals else float("nan")
            mean_sharp = sum(sharpness_vals) / len(sharpness_vals) if sharpness_vals else float("nan")

            results[f"uq/{n_s}/coverage"] = mean_cov
            results[f"uq/{n_s}/sharpness"] = mean_sharp
            logger.info(
                "UQ n_samples=%d  coverage=%.4f  sharpness=%.6f",
                n_s, mean_cov, mean_sharp,
            )

        return results

    # ------------------------------------------------------------------
    # Full ablation suite
    # ------------------------------------------------------------------

    def run_full_ablation(self) -> dict[str, Any]:
        """Execute all ablation experiments and persist results.

        Saves a JSON file to ``<config.output_dir>/ablation_results.json``
        (if ``output_dir`` is configured).

        Returns:
            Comprehensive dict of all ablation results.
        """
        all_results: dict[str, Any] = {}

        logger.info("=== Running full ablation suite ===")

        logger.info("--- Forward-only evaluation ---")
        all_results.update(self.evaluate_forward_only())

        logger.info("--- Without physics loss ---")
        all_results.update(self.evaluate_without_physics_loss())

        logger.info("--- Without consistency loss ---")
        all_results.update(self.evaluate_without_consistency_loss())

        logger.info("--- Without shared weights ---")
        all_results.update(self.evaluate_without_shared_weights())

        logger.info("--- UQ quality vs sample count ---")
        all_results.update(self.evaluate_uq_quality())

        logger.info("--- Full model (baseline) ---")
        baseline = self._evaluate_loop(self.model)
        all_results.update({f"baseline/{k}": v for k, v in baseline.items()})

        # Persist
        output_dir: str | None = OmegaConf.select(
            self.config, "output_dir", default=None
        )
        if output_dir is not None:
            out_path = Path(output_dir) / "ablation_results.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # Convert any non-serialisable values
            serialisable = {
                k: v if isinstance(v, (int, float, str, bool)) else str(v)
                for k, v in all_results.items()
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(serialisable, f, indent=2)
            logger.info("Ablation results saved to %s", out_path)

        logger.info("=== Ablation suite complete ===")
        return all_results
