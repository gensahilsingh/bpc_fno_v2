"""Rectangular-grid smoke-only monodomain solver for Windows validation runs."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class HybridMonodomainSolver:
    """Finite-difference monodomain solver for smoke-only local runs.

    This solver is intentionally limited to local smoke validation. It uses
    Crank-Nicolson diffusion on a regular Cartesian grid and an approximate
    lookup-table ionic callback. The fully coupled production path lives in
    the Linux openCARP backend instead.
    """

    def __init__(
        self,
        conductivity_tensor: np.ndarray,
        grid_shape: tuple[int, int, int],
        voxel_size_cm: float,
        chi_cm_inv: float,
        Cm_uF_per_cm2: float,
        method: str = "direct",
    ) -> None:
        self.grid_shape = tuple(int(v) for v in grid_shape)
        self.nx, self.ny, self.nz = self.grid_shape
        self.n_voxels = self.nx * self.ny * self.nz
        self.h = float(voxel_size_cm)
        self.chi_cm_inv = float(chi_cm_inv)
        self.Cm_uF_per_cm2 = float(Cm_uF_per_cm2)
        self.method = str(method)

        self._D = np.asarray(conductivity_tensor, dtype=np.float64)
        self._A = self._assemble_diffusion_matrix()

        self._dt_ms: float | None = None
        self._lhs: sp.csc_matrix | None = None
        self._rhs_mat: sp.csr_matrix | None = None
        self._direct_solver: spla.SuperLU | None = None
        self._cg_precond: spla.LinearOperator | None = None

    def _flat_index(self, i: int, j: int, k: int) -> int:
        return (i * self.ny + j) * self.nz + k

    def _clip(self, i: int, j: int, k: int) -> tuple[int, int, int]:
        return (
            min(max(i, 0), self.nx - 1),
            min(max(j, 0), self.ny - 1),
            min(max(k, 0), self.nz - 1),
        )

    def _assemble_diffusion_matrix(self) -> sp.csr_matrix:
        nx, ny, nz = self.grid_shape
        h2 = self.h * self.h
        D = self._D

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        shifts = {
            0: (1, 0, 0),
            1: (0, 1, 0),
            2: (0, 0, 1),
        }

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    centre = self._flat_index(i, j, k)

                    for axis, (di, dj, dk) in shifts.items():
                        if i + di < nx and j + dj < ny and k + dk < nz:
                            ni, nj, nk = i + di, j + dj, k + dk
                            coeff = 0.5 * (
                                D[i, j, k, axis, axis]
                                + D[ni, nj, nk, axis, axis]
                            ) / h2
                            rows.extend([centre, centre])
                            cols.extend([centre, self._flat_index(ni, nj, nk)])
                            vals.extend([-coeff, coeff])

                        if i - di >= 0 and j - dj >= 0 and k - dk >= 0:
                            ni, nj, nk = i - di, j - dj, k - dk
                            coeff = 0.5 * (
                                D[i, j, k, axis, axis]
                                + D[ni, nj, nk, axis, axis]
                            ) / h2
                            rows.extend([centre, centre])
                            cols.extend([centre, self._flat_index(ni, nj, nk)])
                            vals.extend([-coeff, coeff])

                    for p in range(3):
                        for q in range(3):
                            if p == q:
                                continue
                            coeff = D[i, j, k, p, q] / (4.0 * h2)
                            if abs(coeff) < 1e-18:
                                continue

                            dip, djp, dkp = shifts[p]
                            diq, djq, dkq = shifts[q]
                            offsets = [
                                (i + dip + diq, j + djp + djq, k + dkp + dkq, +1.0),
                                (i + dip - diq, j + djp - djq, k + dkp - dkq, -1.0),
                                (i - dip + diq, j - djp + djq, k - dkp + dkq, -1.0),
                                (i - dip - diq, j - djp - djq, k - dkp - dkq, +1.0),
                            ]
                            for ii, jj, kk, sign in offsets:
                                ci, cj, ck = self._clip(ii, jj, kk)
                                rows.append(centre)
                                cols.append(self._flat_index(ci, cj, ck))
                                vals.append(sign * coeff)

        matrix = sp.coo_matrix(
            (vals, (rows, cols)),
            shape=(self.n_voxels, self.n_voxels),
            dtype=np.float64,
        ).tocsr()
        matrix.sum_duplicates()
        return matrix

    def set_dt(self, dt_ms: float) -> None:
        if self._dt_ms == dt_ms:
            return

        dt_s = float(dt_ms) * 1e-3
        scale = dt_s / (self.chi_cm_inv * self.Cm_uF_per_cm2 * 1e-6)
        identity = sp.eye(self.n_voxels, format="csr", dtype=np.float64)
        A_dt = self._A * scale
        self._lhs = (identity - 0.5 * A_dt).tocsc()
        self._rhs_mat = identity + 0.5 * A_dt

        if self.method == "cg":
            ilu = spla.spilu(self._lhs, drop_tol=1e-4)
            self._cg_precond = spla.LinearOperator(
                (self.n_voxels, self.n_voxels), ilu.solve
            )
            self._direct_solver = None
        else:
            self._direct_solver = spla.splu(self._lhs)
            self._cg_precond = None

        self._dt_ms = float(dt_ms)

    def _solve_linear(self, rhs: np.ndarray, guess: np.ndarray) -> np.ndarray:
        if self.method == "cg":
            solution, info = spla.cg(
                self._lhs,
                rhs,
                x0=guess,
                M=self._cg_precond,
                rtol=1e-6,
                atol=0.0,
                maxiter=200,
            )
            if info != 0:
                raise RuntimeError(f"CG diffusion solve failed with info={info}.")
            return solution
        return self._direct_solver.solve(rhs)

    def compute_J_i(self, V_field: np.ndarray) -> np.ndarray:
        edge_order = 2 if min(self.grid_shape) >= 3 else 1
        grad_V = np.stack(
            np.gradient(V_field, self.h, edge_order=edge_order), axis=-1
        )
        return -np.einsum("...ab,...b->...a", self._D, grad_V)

    def solve(
        self,
        V_init: np.ndarray,
        ionic_callback: Callable[[np.ndarray, float, float], np.ndarray],
        stimulus_fn: Callable[[float], np.ndarray],
        dt_ms: float,
        total_time_ms: float,
        output_times_ms: np.ndarray,
    ) -> dict[str, np.ndarray]:
        self.set_dt(dt_ms)

        V = np.asarray(V_init, dtype=np.float64).reshape(self.n_voxels).copy()
        n_steps = int(np.round(float(total_time_ms) / float(dt_ms)))
        output_steps = {
            int(np.round(float(t) / float(dt_ms))): out_idx
            for out_idx, t in enumerate(np.asarray(output_times_ms, dtype=np.float64))
        }

        V_snapshots: list[np.ndarray] = []
        J_snapshots: list[np.ndarray] = []
        t_snapshots: list[float] = []

        dt_s = float(dt_ms) * 1e-3
        reaction_scale = dt_s / (self.Cm_uF_per_cm2 * 1e-6)

        for step in range(n_steps + 1):
            t_ms = step * float(dt_ms)

            if step in output_steps:
                V_field = V.reshape(self.grid_shape)
                V_snapshots.append(V_field.astype(np.float32))
                J_snapshots.append(self.compute_J_i(V_field).astype(np.float32))
                t_snapshots.append(t_ms)

            if step == n_steps:
                break

            rhs = self._rhs_mat @ V
            V_half = self._solve_linear(rhs, V)

            i_ion = ionic_callback(V_half, dt_ms, t_ms)
            i_stim = stimulus_fn(t_ms + 0.5 * float(dt_ms))
            V_react = V_half + reaction_scale * (i_stim - i_ion)

            rhs_2 = self._rhs_mat @ V_react
            V = self._solve_linear(rhs_2, V_react)

        return {
            "V_m": np.stack(V_snapshots, axis=0),
            "J_i": np.stack(J_snapshots, axis=0),
            "t_ms": np.asarray(t_snapshots, dtype=np.float32),
        }
