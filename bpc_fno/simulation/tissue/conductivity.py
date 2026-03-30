"""Anisotropic intracellular conductivity tensor for ventricular myocardium.

The monodomain formulation requires the effective intracellular conductivity
tensor **D_i** which, in a fibre-based coordinate system, is given by

    D_i = sigma_it * I  +  (sigma_il - sigma_it) * f f^T

where *f* is the local fibre-direction unit vector, *sigma_il* is the
intracellular longitudinal conductivity, and *sigma_it* is the intracellular
transverse conductivity.

Physics references
------------------
* Clerc J, "Directional differences of impulse spread in trabecular muscle
  from mammalian heart", J Physiol 255(2):335-346, 1976.
* Roberts DE et al., "Effect of tissue anisotropy on extracellular potential
  fields in canine myocardium in situ", Circ Res 44(5):701-712, 1979.
* Potse M et al., "A comparison of monodomain and bidomain reaction-diffusion
  models for action potential propagation in the human heart", IEEE Trans
  Biomed Eng 53(12):2425-2435, 2006.
"""

from __future__ import annotations

import numpy as np


class ConductivityTensor:
    """Build the anisotropic intracellular conductivity tensor field.

    Parameters
    ----------
    sigma_il : float
        Intracellular longitudinal conductivity (S/cm).  Typical value:
        3.0e-3 S/cm.
    sigma_it : float
        Intracellular transverse conductivity (S/cm).  Typical value:
        3.0e-4 S/cm.
    fiber_field : np.ndarray
        Shape ``(N, N, N, 3)`` — unit fibre vectors at every voxel.
    fibrosis_mask : np.ndarray | None
        Shape ``(N, N, N)`` bool — *True* at fibrotic (non-conducting) voxels.
        If *None*, no fibrosis is applied.
    """

    def __init__(
        self,
        sigma_il: float = 3.0e-3,
        sigma_it: float = 3.0e-4,
        fiber_field: np.ndarray = np.empty(0),
        fibrosis_mask: np.ndarray | None = None,
    ) -> None:
        self.sigma_il: float = sigma_il
        self.sigma_it: float = sigma_it
        self.fiber_field: np.ndarray = fiber_field
        self.fibrosis_mask: np.ndarray | None = fibrosis_mask

        if fiber_field.ndim != 4 or fiber_field.shape[-1] != 3:
            raise ValueError(
                f"fiber_field must have shape (N,N,N,3), got {fiber_field.shape}."
            )

    # ------------------------------------------------------------------
    # Forward tensor  D_i = sigma_it * I + (sigma_il - sigma_it) * f f^T
    # ------------------------------------------------------------------

    def get_tensor_field(self) -> np.ndarray:
        """Compute the intracellular conductivity tensor at every voxel.

        .. math::
            D_i(\\mathbf{x}) = \\sigma_{it}\\,\\mathbf{I}
            + (\\sigma_{il} - \\sigma_{it})\\,\\mathbf{f}\\mathbf{f}^T

        At fibrotic voxels the tensor is set to zero (electrical isolation).

        Returns
        -------
        np.ndarray
            Shape ``(N, N, N, 3, 3)`` float64 — conductivity tensor in S/cm.
        """
        f = self.fiber_field  # (N,N,N,3)
        spatial_shape = f.shape[:3]

        # Isotropic part: sigma_it * I  broadcast over spatial dims
        eye = np.zeros((*spatial_shape, 3, 3), dtype=np.float64)
        eye[..., 0, 0] = self.sigma_it
        eye[..., 1, 1] = self.sigma_it
        eye[..., 2, 2] = self.sigma_it

        # Anisotropic part: (sigma_il - sigma_it) * f f^T
        # f f^T via outer product: f[..., i] * f[..., j]
        ff_T = f[..., :, None] * f[..., None, :]  # (N,N,N,3,3)
        D_i = eye + (self.sigma_il - self.sigma_it) * ff_T

        # Zero out fibrotic voxels
        if self.fibrosis_mask is not None:
            D_i[self.fibrosis_mask] = 0.0

        return D_i

    # ------------------------------------------------------------------
    # Inverse tensor  D_i^{-1}
    # ------------------------------------------------------------------

    def get_inverse_tensor_field(self) -> np.ndarray:
        """Compute the inverse of the intracellular conductivity tensor.

        Because *D_i* is constructed from a scalar isotropic part plus a
        rank-1 fibre update, its inverse can be written analytically:

        .. math::
            D_i^{-1} = \\frac{1}{\\sigma_{it}}\\,\\mathbf{I}
            + \\left(\\frac{1}{\\sigma_{il}} - \\frac{1}{\\sigma_{it}}\\right)
            \\,\\mathbf{f}\\mathbf{f}^T

        At fibrotic voxels the inverse is set to zero.

        Returns
        -------
        np.ndarray
            Shape ``(N, N, N, 3, 3)`` float64 — inverse conductivity in
            (S/cm)^{-1}.
        """
        f = self.fiber_field
        spatial_shape = f.shape[:3]

        inv_sigma_it = 1.0 / self.sigma_it
        inv_sigma_il = 1.0 / self.sigma_il

        eye = np.zeros((*spatial_shape, 3, 3), dtype=np.float64)
        eye[..., 0, 0] = inv_sigma_it
        eye[..., 1, 1] = inv_sigma_it
        eye[..., 2, 2] = inv_sigma_it

        ff_T = f[..., :, None] * f[..., None, :]
        D_inv = eye + (inv_sigma_il - inv_sigma_it) * ff_T

        if self.fibrosis_mask is not None:
            D_inv[self.fibrosis_mask] = 0.0

        return D_inv
