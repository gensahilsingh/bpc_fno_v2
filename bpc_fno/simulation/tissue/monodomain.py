"""Monodomain PDE solver with operator-splitting (Crank-Nicolson + Rush-Larsen).

The monodomain equation reads:

.. math::
    \\beta\\,C_m\\,\\frac{\\partial V_m}{\\partial t}
    = \\nabla \\cdot (D_i\\,\\nabla V_m) - \\beta\\,I_{ion}(V_m, w) + I_{stim}

where *V_m* is the transmembrane potential, *D_i* is the effective
intracellular conductivity tensor, *beta* is the membrane surface-to-volume
ratio, *C_m* is the specific membrane capacitance, *I_ion* is the total
ionic current (from the TT2006 cell model, for example), and *I_stim* is
the externally applied stimulus current.

Operator-splitting strategy
---------------------------
1. **Diffusion half-step** (PDE) — Crank-Nicolson implicit/explicit scheme:
   ``(I - dt/2 * A) V^{n+1} = (I + dt/2 * A) V^n``
   with ``A = L / (beta * C_m)``, *L* the discrete anisotropic Laplacian.
2. **Reaction step** (ODE) — delegated to an external ionic-model callback
   which is expected to implement the Rush-Larsen exponential integrator for
   gating variables internally (e.g. via Myokit).

Boundary conditions: zero-flux Neumann (dV_m/dn = 0) on all faces,
implemented by ghost-node mirroring in the finite-difference stencil.

Physics references
------------------
* ten Tusscher KHWJ & Panfilov AV, "Alternans and spiral breakup in a
  human ventricular tissue model", Am J Physiol Heart Circ Physiol
  291:H1088-H1100, 2006.
* Rush S & Larsen H, "A practical algorithm for solving dynamic membrane
  equations", IEEE Trans Biomed Eng 25(4):389-392, 1978.
* Qu Z & Garfinkel A, "An advanced algorithm for solving partial
  differential equation in cardiac conduction", IEEE Trans Biomed Eng
  46(9):1166-1168, 1999.
* Sundnes J et al., "Computing the Electrical Activity in the Heart",
  Springer, 2006.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from bpc_fno.simulation.tissue.conductivity import ConductivityTensor
from bpc_fno.simulation.tissue.geometry import VentricularSlab


class MonodomainSolver:
    """Finite-difference monodomain solver on a regular Cartesian grid.

    Parameters
    ----------
    slab : VentricularSlab
        Geometry provider (grid size, voxel spacing, fibre field, etc.).
    config : DictConfig
        Hydra / OmegaConf configuration object.  Expected keys used
        internally:

        * ``config.beta``   — surface-to-volume ratio (cm^{-1}), default 0.14
        * ``config.C_m``    — specific membrane capacitance (uF/cm^2), default 1.0
        * ``config.stim_amplitude`` — stimulus amplitude (uA/cm^2), default -52.0
        * ``config.stim_duration_ms`` — stimulus duration (ms), default 2.0
    """

    # Default electrophysiology constants
    _DEFAULT_BETA: float = 0.14        # cm^{-1}
    _DEFAULT_C_M: float = 1.0          # uF/cm^2
    _DEFAULT_STIM_AMP: float = -52.0   # uA/cm^2
    _DEFAULT_STIM_DUR: float = 2.0     # ms
    _V_REST: float = -85.23            # mV  (TT2006 resting potential)

    def __init__(self, slab: VentricularSlab, config: Any) -> None:
        self.slab: VentricularSlab = slab
        self.config = config

        self.beta: float = getattr(config, "beta", self._DEFAULT_BETA)
        self.C_m: float = getattr(config, "C_m", self._DEFAULT_C_M)
        self.stim_amplitude: float = getattr(
            config, "stim_amplitude", self._DEFAULT_STIM_AMP
        )
        self.stim_duration_ms: float = getattr(
            config, "stim_duration_ms", self._DEFAULT_STIM_DUR
        )

        # Solver method: 'explicit' (fast), 'cg' (moderate), 'direct' (slow)
        # Check config.monodomain.solver_method first, then config.solver_method
        _sm = "explicit"
        if hasattr(config, "monodomain") and hasattr(config.monodomain, "solver_method"):
            _sm = str(config.monodomain.solver_method)
        elif hasattr(config, "solver_method"):
            _sm = str(config.solver_method)
        self.solver_method: str = _sm

        self.N: int = slab.grid_size
        self.h: float = slab.voxel_size_cm
        self.n_voxels: int = self.N ** 3

        # State arrays — initialised in setup()
        self.V_m: np.ndarray | None = None
        self.conductivity: ConductivityTensor | None = None
        self._D_tensor: np.ndarray | None = None

        # Sparse operators — built in setup()
        self._L: sp.csr_matrix | None = None       # discrete Laplacian
        self._A: sp.csr_matrix | None = None        # L / (beta * C_m)
        self._cn_lhs: sp.csr_matrix | None = None   # Crank-Nicolson LHS
        self._cn_rhs: sp.csr_matrix | None = None   # Crank-Nicolson RHS
        self._cg_precond: Any = None                 # ILU preconditioner for CG

        # Pacing
        self._pacing_voxel: tuple[int, int, int] | None = None
        self._stim_active: bool = False
        self._stim_start_ms: float = 0.0

    # ==================================================================
    # Setup
    # ==================================================================

    def setup(
        self,
        conductivity: ConductivityTensor,
        initial_states: dict[str, Any] | None = None,
        dt_ms: float = 0.02,
    ) -> None:
        """Assemble diffusion operators and initialise state arrays.

        Parameters
        ----------
        conductivity : ConductivityTensor
            Provides the tensor field *D_i* and optional fibrosis mask.
        initial_states : dict | None
            Optional dictionary of pre-computed state arrays.  If *None*,
            V_m is initialised to the TT2006 resting potential everywhere.
        dt_ms : float
            Time step in milliseconds used to pre-factor the Crank-Nicolson
            matrices.  Default 0.02 ms.
        """
        self.conductivity = conductivity
        self._D_tensor = conductivity.get_tensor_field()  # (N,N,N,3,3)

        # Initialise V_m
        if initial_states is not None and "V_m" in initial_states:
            self.V_m = initial_states["V_m"].ravel().astype(np.float64).copy()
        else:
            self.V_m = np.full(self.n_voxels, self._V_REST, dtype=np.float64)

        # Build discrete Laplacian  L  (sparse CSR)
        self._L = self._build_laplacian()

        # A = L / (beta * C_m)
        # Units: L in S/cm^3, beta*C_m in uF/cm^3 → A in S/uF = 1e6/s = 1e3/ms
        # dt*A is dimensionless when dt is in ms
        self._A = self._L / (self.beta * self.C_m)

        if self.solver_method == "explicit":
            # Forward Euler diffusion: V_new = V_old + dt * A @ V_old
            # Stability: dt <= 1 / spectral_radius(A) ≈ h²/(2*d*D_max/(beta*C_m))
            # For safety, we just check at runtime and warn.
            D_max = float(np.max(np.abs(self._D_tensor)))
            d = 3  # spatial dimensions
            dt_stability = (self.h ** 2 * self.beta * self.C_m) / (2 * d * D_max) if D_max > 0 else 1e6
            if dt_ms > dt_stability:
                import logging
                logging.getLogger(__name__).warning(
                    "Explicit solver: dt=%.4f ms exceeds stability limit %.4f ms "
                    "(D_max=%.2e). Simulation may be unstable.",
                    dt_ms, dt_stability, D_max,
                )
        elif self.solver_method == "cg":
            # Crank-Nicolson with CG solver + ILU preconditioner
            alpha = dt_ms / 2.0
            I_sp = sp.eye(self.n_voxels, format="csr")
            self._cn_lhs = (I_sp - alpha * self._A).tocsc()
            self._cn_rhs = I_sp + alpha * self._A
            # Build ILU preconditioner once
            ilu = spla.spilu(self._cn_lhs, drop_tol=1e-4)
            self._cg_precond = spla.LinearOperator(
                (self.n_voxels, self.n_voxels), ilu.solve
            )
        else:  # 'direct'
            alpha = dt_ms / 2.0
            I_sp = sp.eye(self.n_voxels, format="csr")
            self._cn_lhs = I_sp - alpha * self._A
            self._cn_rhs = I_sp + alpha * self._A

    # ------------------------------------------------------------------
    # Sparse Laplacian assembly
    # ------------------------------------------------------------------

    def _build_laplacian(self) -> sp.csr_matrix:
        """Assemble the anisotropic discrete Laplacian operator.

        For the full anisotropic diffusion operator
        ``div(D_i grad V_m)`` on a uniform grid with spacing *h*, we use a
        second-order central finite-difference stencil.  For each spatial
        direction pair (p, q) the contribution is:

        * **Diagonal terms** (p == q):
          ``d/dx_p [ D_{pp} dV/dx_p ] ≈
            (D_{pp}^{+} (V_{+} - V) - D_{pp}^{-} (V - V_{-})) / h^2``
          where ``D_{pp}^{+/-}`` is the conductivity at the half-voxel face.

        * **Off-diagonal / cross terms** (p != q):
          ``d/dx_p [ D_{pq} dV/dx_q ]``  discretised with a compact
          4-neighbour stencil on the (p,q)-plane.  These contribute to a
          wider stencil but are included for correctness.

        Zero-flux Neumann BCs are enforced by clamping neighbour indices to
        the grid boundary (ghost-node mirroring).

        Returns
        -------
        scipy.sparse.csr_matrix
            Shape ``(n_voxels, n_voxels)`` — the discrete Laplacian *L*.
        """
        N = self.N
        h = self.h
        D = self._D_tensor  # (N,N,N,3,3)

        # Mapping from (i,j,k) to flat index
        def idx(i: np.ndarray, j: np.ndarray, k: np.ndarray) -> np.ndarray:
            # Clamp for Neumann BC (ghost-node mirror)
            ic = np.clip(i, 0, N - 1)
            jc = np.clip(j, 0, N - 1)
            kc = np.clip(k, 0, N - 1)
            return ic * N * N + jc * N + kc

        # Flat arrays of all voxel indices
        ii, jj, kk = np.mgrid[0:N, 0:N, 0:N]
        ii_f = ii.ravel()
        jj_f = jj.ravel()
        kk_f = kk.ravel()
        centre_idx = idx(ii_f, jj_f, kk_f)

        rows_list: list[np.ndarray] = []
        cols_list: list[np.ndarray] = []
        vals_list: list[np.ndarray] = []

        h2 = h * h

        # Axis direction vectors for shift operations
        shifts = {
            0: (1, 0, 0),
            1: (0, 1, 0),
            2: (0, 0, 1),
        }

        # ----------------------------------------------------------
        # Diagonal diffusion terms: d/dx_p [D_{pp} dV/dx_p]
        # ----------------------------------------------------------
        for p in range(3):
            di, dj, dk = shifts[p]

            # D at half-grid points: D_plus = 0.5*(D[here] + D[here+dp])
            # Neighbour indices (clamped)
            ip = np.clip(ii_f + di, 0, N - 1)
            jp = np.clip(jj_f + dj, 0, N - 1)
            kp = np.clip(kk_f + dk, 0, N - 1)

            im = np.clip(ii_f - di, 0, N - 1)
            jm = np.clip(jj_f - dj, 0, N - 1)
            km = np.clip(kk_f - dk, 0, N - 1)

            D_pp_here = D[ii_f, jj_f, kk_f, p, p]
            D_pp_plus = D[ip, jp, kp, p, p]
            D_pp_minus = D[im, jm, km, p, p]

            # Face conductivities (harmonic-like averaging via arithmetic mean)
            D_face_plus = 0.5 * (D_pp_here + D_pp_plus)
            D_face_minus = 0.5 * (D_pp_here + D_pp_minus)

            coeff_plus = D_face_plus / h2
            coeff_minus = D_face_minus / h2
            coeff_centre = -(coeff_plus + coeff_minus)

            # Plus neighbour
            nbr_plus = idx(ii_f + di, jj_f + dj, kk_f + dk)
            rows_list.append(centre_idx)
            cols_list.append(nbr_plus)
            vals_list.append(coeff_plus)

            # Minus neighbour
            nbr_minus = idx(ii_f - di, jj_f - dj, kk_f - dk)
            rows_list.append(centre_idx)
            cols_list.append(nbr_minus)
            vals_list.append(coeff_minus)

            # Centre contribution
            rows_list.append(centre_idx)
            cols_list.append(centre_idx)
            vals_list.append(coeff_centre)

        # ----------------------------------------------------------
        # Off-diagonal (cross) terms: d/dx_p [D_{pq} dV/dx_q], p!=q
        # Using the compact 4-point stencil:
        #   d/dx_p [D_{pq} dV/dx_q] ≈
        #     D_{pq} * (V_{+p,+q} - V_{+p,-q} - V_{-p,+q} + V_{-p,-q})
        #     / (4 h^2)
        # with D_{pq} evaluated at the centre voxel.
        # ----------------------------------------------------------
        for p in range(3):
            for q in range(3):
                if p == q:
                    continue
                dip, djp, dkp = shifts[p]
                diq, djq, dkq = shifts[q]

                D_pq = D[ii_f, jj_f, kk_f, p, q]
                coeff = D_pq / (4.0 * h2)

                # +p, +q
                rows_list.append(centre_idx)
                cols_list.append(idx(ii_f + dip + diq, jj_f + djp + djq, kk_f + dkp + dkq))
                vals_list.append(coeff)

                # +p, -q
                rows_list.append(centre_idx)
                cols_list.append(idx(ii_f + dip - diq, jj_f + djp - djq, kk_f + dkp - dkq))
                vals_list.append(-coeff)

                # -p, +q
                rows_list.append(centre_idx)
                cols_list.append(idx(ii_f - dip + diq, jj_f - djp + djq, kk_f - dkp + dkq))
                vals_list.append(-coeff)

                # -p, -q
                rows_list.append(centre_idx)
                cols_list.append(idx(ii_f - dip - diq, jj_f - djp - djq, kk_f - dkp - dkq))
                vals_list.append(coeff)

        # Assemble sparse matrix
        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        vals = np.concatenate(vals_list)

        L = sp.coo_matrix((vals, (rows, cols)), shape=(self.n_voxels, self.n_voxels))
        L = L.tocsr()
        # Sum duplicate entries (from overlapping stencil contributions)
        L.sum_duplicates()

        return L

    # ==================================================================
    # Pacing
    # ==================================================================

    def set_pacing_site(self, voxel_idx: tuple[int, int, int]) -> None:
        """Define the stimulus site.

        Parameters
        ----------
        voxel_idx : tuple[int, int, int]
            ``(i, j, k)`` index of the voxel that receives the pacing
            stimulus (rectangular current pulse).
        """
        self._pacing_voxel = voxel_idx

    def _get_I_stim(self, t_ms: float) -> np.ndarray:
        """Return the stimulus current vector at time *t_ms*.

        The stimulus is a rectangular pulse of amplitude
        ``stim_amplitude`` (uA/cm^2) lasting ``stim_duration_ms``,
        applied at the designated pacing voxel.

        Returns
        -------
        np.ndarray
            Shape ``(n_voxels,)`` — stimulus current in uA/cm^2.
        """
        I_stim = np.zeros(self.n_voxels, dtype=np.float64)
        if self._pacing_voxel is None:
            return I_stim

        if self._stim_active and t_ms < (self._stim_start_ms + self.stim_duration_ms):
            i, j, k = self._pacing_voxel
            flat = i * self.N * self.N + j * self.N + k
            I_stim[flat] = self.stim_amplitude
        return I_stim

    # ==================================================================
    # Time stepping
    # ==================================================================

    def step(self, dt_ms: float, I_ion: np.ndarray, t_ms: float) -> None:
        """Advance V_m by one time step using operator splitting.

        Solver methods:
        - ``'explicit'``: Forward Euler diffusion (fast, conditionally stable)
        - ``'cg'``: Crank-Nicolson with CG + ILU preconditioner (moderate)
        - ``'direct'``: Crank-Nicolson with direct sparse solve (slow, robust)

        Parameters
        ----------
        dt_ms : float
            Time step in milliseconds.
        I_ion : np.ndarray
            Ionic current density at each voxel (uA/cm^2), shape
            ``(n_voxels,)`` or ``(N, N, N)``.
        t_ms : float
            Current simulation time in milliseconds.
        """
        assert self.V_m is not None, "Call setup() before step()."

        I_ion_flat = I_ion.ravel()

        # 1. Diffusion step
        if self.solver_method == "explicit":
            # Forward Euler: V_new = V_old + dt * A @ V_old
            self.V_m = self.V_m + dt_ms * self._A.dot(self.V_m)
        elif self.solver_method == "cg":
            # Crank-Nicolson with CG
            rhs = self._cn_rhs.dot(self.V_m)
            self.V_m, info = spla.cg(
                self._cn_lhs, rhs,
                x0=self.V_m,
                M=self._cg_precond,
                tol=1e-6,
                maxiter=100,
            )
            if info != 0:
                import logging
                logging.getLogger(__name__).warning(
                    "CG did not converge at t=%.2f ms (info=%d)", t_ms, info
                )
        else:  # 'direct'
            rhs = self._cn_rhs.dot(self.V_m)
            self.V_m = spla.spsolve(self._cn_lhs, rhs)

        # 2. Reaction step: forward-Euler update for V_m
        I_stim = self._get_I_stim(t_ms)
        self.V_m += dt_ms / self.C_m * (-I_ion_flat + I_stim)

    # ==================================================================
    # Main run loop
    # ==================================================================

    def run(
        self,
        total_time_ms: float,
        dt_ms: float,
        output_stride: int,
        ionic_model_func: Callable[[np.ndarray, float], tuple[np.ndarray, Any]],
    ) -> dict[str, np.ndarray]:
        """Execute the full simulation time loop.

        Parameters
        ----------
        total_time_ms : float
            Total simulation duration in milliseconds.
        dt_ms : float
            Time step size in milliseconds.
        output_stride : int
            Save a snapshot every *output_stride* time steps.
        ionic_model_func : callable
            ``(V_m_flat, dt_ms) -> (I_ion_flat, updated_states)``
            The ionic-model callback.  It receives the current V_m as a
            flat array and the time step, and returns the total ionic
            current and (optionally) updated internal state.  The
            Rush-Larsen exponential integrator for gating variables is
            expected to be implemented inside this callback.

        Returns
        -------
        dict[str, np.ndarray]
            * ``'V_m'``  — shape ``(T, N, N, N)`` transmembrane potential (mV)
            * ``'J_i'``  — shape ``(T, N, N, N, 3)`` intracellular current
              density (uA/cm^2)
            * ``'t_ms'`` — shape ``(T,)`` snapshot times
        """
        assert self.V_m is not None, "Call setup() before run()."

        n_steps = int(np.ceil(total_time_ms / dt_ms))
        N = self.N

        # Pre-allocate output buffers (list, concatenated at the end)
        V_snapshots: list[np.ndarray] = []
        J_snapshots: list[np.ndarray] = []
        t_snapshots: list[float] = []

        # Activate stimulus at t = 0
        self._stim_active = True
        self._stim_start_ms = 0.0

        for step_idx in range(n_steps):
            t_ms_now = step_idx * dt_ms

            # --- Ionic model (reaction / Rush-Larsen for gating vars) ---
            I_ion_flat, _ = ionic_model_func(self.V_m, dt_ms)

            # --- PDE + reaction step ---
            self.step(dt_ms, I_ion_flat, t_ms_now)

            # --- Snapshot ---
            if step_idx % output_stride == 0:
                V_field = self.V_m.reshape(N, N, N).copy()
                V_snapshots.append(V_field)
                J_snapshots.append(self.compute_J_i(V_field))
                t_snapshots.append(t_ms_now)

        return {
            "V_m": np.stack(V_snapshots, axis=0),
            "J_i": np.stack(J_snapshots, axis=0),
            "t_ms": np.array(t_snapshots, dtype=np.float64),
        }

    # ==================================================================
    # Intracellular current density
    # ==================================================================

    def compute_J_i(self, V_m_field: np.ndarray) -> np.ndarray:
        """Compute the intracellular current density J_i = -D_i * grad(V_m).

        The gradient is evaluated using second-order central finite
        differences with zero-flux (Neumann) boundary handling.

        Parameters
        ----------
        V_m_field : np.ndarray
            Shape ``(N, N, N)`` — transmembrane potential in mV.

        Returns
        -------
        np.ndarray
            Shape ``(N, N, N, 3)`` — intracellular current density in
            uA/cm^2.
        """
        assert self._D_tensor is not None, "Call setup() before compute_J_i()."

        N = self.N
        h = self.h
        D = self._D_tensor  # (N,N,N,3,3)

        # Central-difference gradient  (zero-flux BC via np.gradient default)
        # np.gradient returns list of arrays, one per axis
        grad_V = np.stack(np.gradient(V_m_field, h, edge_order=2), axis=-1)  # (N,N,N,3)

        # J_i = -D_i @ grad(V_m)  at each voxel
        # D is (N,N,N,3,3), grad_V is (N,N,N,3) -> einsum 'ijkab,ijkb->ijka'
        J_i: np.ndarray = -np.einsum("...ab,...b->...a", D, grad_V)

        return J_i
