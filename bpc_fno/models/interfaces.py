"""Abstract base classes defining the interfaces for BPC-FNO model components."""

from abc import ABC, abstractmethod

from torch import Tensor


class GeometryEncoderInterface(ABC):
    @abstractmethod
    def encode(self, geometry: Tensor) -> Tensor:
        """Encode voxelised geometry channels into a latent feature volume.

        Args:
            geometry: (B, 4, N, N, N) — SDF, mu_r, sigma, mask.

        Returns:
            encoded: (B, C, N, N, N)
        """
        ...


class FNOBackboneInterface(ABC):
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Apply Fourier neural operator spectral blocks.

        Args:
            x: (B, C_in, N, N, N)

        Returns:
            out: (B, C_out, N, N, N)
        """
        ...


class ForwardOperatorInterface(ABC):
    @abstractmethod
    def predict_B(self, J_i: Tensor, geometry: Tensor) -> Tensor:
        """Predict sensor magnetic field from impressed current density.

        Args:
            J_i: (B, 3, N, N, N)
            geometry: (B, 4, N, N, N)

        Returns:
            B_pred: (B, N_sensors*3, T)
        """
        ...


class InverseEncoderInterface(ABC):
    @abstractmethod
    def encode_to_latent(
        self, B_obs: Tensor, geometry: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Encode observed B-field measurements into a VAE latent space.

        Args:
            B_obs: (B, N_sensors*3, T)
            geometry: (B, 4, N, N, N)

        Returns:
            (mu, log_var): each (B, D)
        """
        ...


class DecoderInterface(ABC):
    @abstractmethod
    def decode(self, z: Tensor, geometry: Tensor) -> Tensor:
        """Decode a latent vector into an impressed current density volume.

        Args:
            z: (B, D)
            geometry: (B, 4, N, N, N)

        Returns:
            J_i_hat: (B, 3, N, N, N)
        """
        ...
