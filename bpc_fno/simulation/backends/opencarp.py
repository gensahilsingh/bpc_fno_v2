"""Linux openCARP backend for production monodomain data generation."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bpc_fno.simulation.backends.base import (
    SimulationBackend,
    SimulationContext,
    SimulationResult,
)

logger = logging.getLogger(__name__)


def _s_per_cm_to_s_per_m(value: float) -> float:
    return float(value) * 100.0


def _format_im_param(
    cell_flag: str,
    params: dict[str, object],
) -> str:
    conductance_scales = dict(params.get("conductance_scales", {}))
    ko_mM = float(params.get("ko_mM", 5.4))
    items = [
        f"flags={cell_flag}",
        f"Ko={ko_mM:.6g}",
    ]
    name_map = {
        "I_Na": "GNa",
        "I_CaL": "GCaL",
        "I_Kr": "GKr",
        "I_Ks": "GKs",
    }
    for key, scale in conductance_scales.items():
        if key in name_map:
            items.append(f"{name_map[key]}*{float(scale):.6g}")
    return ",".join(items)


def _parse_igb(path: Path) -> tuple[np.ndarray, dict[str, str]]:
    """Read a scalar IGB file into ``(T, N)`` float64 data."""
    with open(path, "rb") as fh:
        header_bytes = fh.read(1024)
        payload = fh.read()

    header = (
        header_bytes.decode("ascii", errors="ignore")
        .replace("\x00", " ")
        .replace("\x0c", " ")
        .strip()
    )
    matches = re.finditer(
        r"([A-Za-z_ ]+?):(.*?)(?=(?:\s+[A-Za-z_ ][A-Za-z0-9_ ]*?:)|$)",
        header,
    )
    meta = {
        match.group(1).strip().lower().replace(" ", "_"): match.group(2).strip()
        for match in matches
    }

    n_x = int(meta.get("x", "0"))
    n_y = int(meta.get("y", "1"))
    n_z = int(meta.get("z", "1"))
    n_t = int(meta.get("t", "1"))
    dtype_name = meta.get("type", "float").lower()
    system = meta.get("systeme", "little endian").lower()
    endian = "<" if "little" in system else ">"

    if dtype_name == "double":
        dtype = np.dtype(f"{endian}f8")
    else:
        dtype = np.dtype(f"{endian}f4")

    n_values = n_x * n_y * n_z * n_t
    data = np.frombuffer(payload[: n_values * dtype.itemsize], dtype=dtype, count=n_values)
    if data.size != n_values:
        raise ValueError(
            f"IGB file {path} ended early: expected {n_values} values, got {data.size}."
        )
    return data.reshape(n_t, n_x * n_y * n_z).astype(np.float64), meta


def _node_index(
    i: int,
    j: int,
    k: int,
    ny_nodes: int,
    nz_nodes: int,
) -> int:
    return (i * ny_nodes + j) * nz_nodes + k


def _write_regular_hexa_mesh(
    mesh_base: Path,
    voxel_size_cm: float,
    fiber: np.ndarray,
    cell_type_map: np.ndarray,
    fibrosis_mask: np.ndarray,
) -> dict[str, object]:
    """Write a regular hexahedral openCARP mesh matching the voxel grid."""
    nx, ny, nz, _ = fiber.shape
    ny_nodes = ny + 1
    nz_nodes = nz + 1
    h_um = float(voxel_size_cm) * 1.0e4

    pts_path = mesh_base.with_suffix(".pts")
    elem_path = mesh_base.with_suffix(".elem")
    lon_path = mesh_base.with_suffix(".lon")

    n_nodes = (nx + 1) * (ny + 1) * (nz + 1)
    with open(pts_path, "w", encoding="utf-8") as fp:
        fp.write(f"{n_nodes}\n")
        for i in range(nx + 1):
            for j in range(ny + 1):
                for k in range(nz + 1):
                    fp.write(
                        f"{i * h_um:.6f} {j * h_um:.6f} {k * h_um:.6f}\n"
                    )

    element_tags = np.zeros((nx, ny, nz), dtype=np.int32)
    n_elems = nx * ny * nz
    with open(elem_path, "w", encoding="utf-8") as fp:
        fp.write(f"{n_elems}\n")
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    n000 = _node_index(i, j, k, ny_nodes, nz_nodes)
                    n100 = _node_index(i + 1, j, k, ny_nodes, nz_nodes)
                    n010 = _node_index(i, j + 1, k, ny_nodes, nz_nodes)
                    n110 = _node_index(i + 1, j + 1, k, ny_nodes, nz_nodes)
                    n001 = _node_index(i, j, k + 1, ny_nodes, nz_nodes)
                    n101 = _node_index(i + 1, j, k + 1, ny_nodes, nz_nodes)
                    n011 = _node_index(i, j + 1, k + 1, ny_nodes, nz_nodes)
                    n111 = _node_index(i + 1, j + 1, k + 1, ny_nodes, nz_nodes)

                    cell_type = int(cell_type_map[i, j, k])
                    fibrotic = bool(fibrosis_mask[i, j, k])
                    tag = {0: 1, 1: 2, 2: 3}[cell_type] + (3 if fibrotic else 0)
                    element_tags[i, j, k] = tag

                    fp.write(
                        "Hx "
                        f"{n000} {n100} {n010} {n110} "
                        f"{n001} {n101} {n011} {n111} "
                        f"{tag}\n"
                    )

    with open(lon_path, "w", encoding="utf-8") as fp:
        fp.write("1\n")
        for vec in fiber.reshape(-1, 3):
            fp.write(f"{vec[0]:.8f} {vec[1]:.8f} {vec[2]:.8f}\n")

    return {
        "pts_path": pts_path,
        "elem_path": elem_path,
        "lon_path": lon_path,
        "element_tags": element_tags,
    }


def _box_from_pacing_site(
    pacing_site_voxel: tuple[int, int, int],
    voxel_size_cm: float,
    radius_cm: float,
) -> tuple[np.ndarray, np.ndarray]:
    h_um = float(voxel_size_cm) * 1.0e4
    r_um = float(radius_cm) * 1.0e4
    centre = (np.asarray(pacing_site_voxel, dtype=np.float64) + 0.5) * h_um
    return centre - r_um, centre + r_um


def _stimulus_mask_from_box(
    grid_shape: tuple[int, int, int],
    voxel_size_cm: float,
    p0_um: np.ndarray,
    p1_um: np.ndarray,
) -> np.ndarray:
    """Build a voxel mask matching the openCARP stimulus electrode box."""
    h_um = float(voxel_size_cm) * 1.0e4
    lower = np.minimum(p0_um, p1_um)
    upper = np.maximum(p0_um, p1_um)

    i_idx, j_idx, k_idx = np.indices(grid_shape, dtype=np.float64)
    centres_um = np.stack((i_idx + 0.5, j_idx + 0.5, k_idx + 0.5), axis=-1) * h_um
    inside = np.logical_and(centres_um >= lower, centres_um <= upper)
    return np.all(inside, axis=-1).astype(np.uint8)


def _voltage_nodes_to_voxels(
    vm_nodes: np.ndarray,
    grid_shape: tuple[int, int, int],
) -> np.ndarray:
    """Convert node-based voltage output to voxel-centred values."""
    nx, ny, nz = grid_shape
    vm_4d = vm_nodes.reshape(vm_nodes.shape[0], nx + 1, ny + 1, nz + 1)
    return (
        vm_4d[:, :-1, :-1, :-1]
        + vm_4d[:, 1:, :-1, :-1]
        + vm_4d[:, :-1, 1:, :-1]
        + vm_4d[:, 1:, 1:, :-1]
        + vm_4d[:, :-1, :-1, 1:]
        + vm_4d[:, 1:, :-1, 1:]
        + vm_4d[:, :-1, 1:, 1:]
        + vm_4d[:, 1:, 1:, 1:]
    ) / 8.0


def _compute_activation_times(
    vm_voxels: np.ndarray,
    t_ms: np.ndarray,
    threshold_mV: float,
) -> np.ndarray:
    activation = np.full(vm_voxels.shape[1:], np.inf, dtype=np.float64)
    for i in range(vm_voxels.shape[1]):
        for j in range(vm_voxels.shape[2]):
            for k in range(vm_voxels.shape[3]):
                trace = vm_voxels[:, i, j, k]
                idx = np.flatnonzero(trace >= threshold_mV)
                if idx.size:
                    activation[i, j, k] = float(t_ms[int(idx[0])])
    return activation


@dataclass(slots=True)
class _BaseCarpAdapter:
    binary_name: str

    def is_available(self) -> bool:
        return shutil.which(self.binary_name) is not None

    def _build_parameter_file(
        self,
        context: SimulationContext,
        mesh_base: Path,
        sim_dir: Path,
    ) -> Path:
        mono_cfg = context.config.monodomain
        tissue_cfg = context.config.tissue
        ionic_cfg = context.config.ionic

        stim_p0_um, stim_p1_um = _box_from_pacing_site(
            context.pacing_site_voxel,
            context.slab.voxel_size_cm,
            float(mono_cfg.stimulus_radius_cm),
        )
        dt_out = float(context.output_times_ms[1] - context.output_times_ms[0])
        dt_us = int(round(float(mono_cfg.dt_ms) * 1000.0))

        healthy_g_il = _s_per_cm_to_s_per_m(float(tissue_cfg.sigma_il))
        healthy_g_it = _s_per_cm_to_s_per_m(float(tissue_cfg.sigma_it))
        fibrotic_scale = float(getattr(mono_cfg, "fibrosis_conductivity_scale", 0.0))
        fibrotic_g_il = healthy_g_il * fibrotic_scale
        fibrotic_g_it = healthy_g_it * fibrotic_scale

        cell_region_defs = [
            ("ENDO", "endo_healthy", 1),
            ("MCELL", "mid_healthy", 2),
            ("EPI", "epi_healthy", 3),
            ("ENDO", "endo_fibrotic", 4),
            ("MCELL", "mid_fibrotic", 5),
            ("EPI", "epi_fibrotic", 6),
        ]

        lines = [
            f'simID = "{sim_dir.as_posix()}"',
            f'meshname = "{mesh_base.as_posix()}"',
            f"dt = {dt_us}",
            f"tend = {float(mono_cfg.total_time_ms):.6f}",
            f"spacedt = {dt_out:.6f}",
            f"timedt = {max(dt_out, 1.0):.6f}",
            'vofile = "vm"',
            "gridout_i = 2",
            "dataout_i = 2",
            f"prepacing_beats = {int(getattr(ionic_cfg, 'n_prepacing_beats', 0))}",
            f"prepacing_bcl = {float(context.params['pacing_cl_ms']):.6f}",
            f"prepacing_stimdur = {float(mono_cfg.stimulus_duration_ms):.6f}",
            f"prepacing_stimstr = {float(mono_cfg.opencarp_stimulus_strength):.6f}",
            f"num_imp_regions = {len(cell_region_defs)}",
        ]

        for idx, (flag, name, region_id) in enumerate(cell_region_defs):
            lines.extend(
                [
                    f'imp_region[{idx}].name = "{name}"',
                    f"imp_region[{idx}].im = tenTusscherPanfilov",
                    f'imp_region[{idx}].im_param = "{_format_im_param(flag, dict(context.params))}"',
                    "imp_region[{idx}].cellSurfVolRatio = 0.14".format(idx=idx),
                    "imp_region[{idx}].volFrac = 1".format(idx=idx),
                    f"imp_region[{idx}].num_IDs = 1",
                    f"imp_region[{idx}].ID = {region_id}",
                ]
            )

        gregions = [
            ("healthy_endo", 1, healthy_g_il, healthy_g_it),
            ("healthy_mid", 2, healthy_g_il, healthy_g_it),
            ("healthy_epi", 3, healthy_g_il, healthy_g_it),
            ("fibrotic_endo", 4, fibrotic_g_il, fibrotic_g_it),
            ("fibrotic_mid", 5, fibrotic_g_il, fibrotic_g_it),
            ("fibrotic_epi", 6, fibrotic_g_il, fibrotic_g_it),
        ]
        lines.append(f"num_gregions = {len(gregions)}")
        for idx, (name, region_id, g_il, g_it) in enumerate(gregions):
            lines.extend(
                [
                    f'gregion[{idx}].name = "{name}"',
                    f"gregion[{idx}].g_il = {g_il:.8f}",
                    f"gregion[{idx}].g_it = {g_it:.8f}",
                    f"gregion[{idx}].g_in = {g_it:.8f}",
                    f"gregion[{idx}].ID = {region_id}",
                    "gregion[{idx}].num_IDs = 1".format(idx=idx),
                ]
            )

        lines.extend(
            [
                "num_stim = 1",
                "stim[0].elec.geom_type = 2",
                f"stim[0].elec.p0[0] = {stim_p0_um[0]:.6f}",
                f"stim[0].elec.p0[1] = {stim_p0_um[1]:.6f}",
                f"stim[0].elec.p0[2] = {stim_p0_um[2]:.6f}",
                f"stim[0].elec.p1[0] = {stim_p1_um[0]:.6f}",
                f"stim[0].elec.p1[1] = {stim_p1_um[1]:.6f}",
                f"stim[0].elec.p1[2] = {stim_p1_um[2]:.6f}",
                f"stim[0].pulse.start = {float(mono_cfg.stimulus_start_ms):.6f}",
                f"stim[0].ptcl.start = {float(mono_cfg.stimulus_start_ms):.6f}",
                f"stim[0].ptcl.duration = {float(mono_cfg.stimulus_duration_ms):.6f}",
                f"stim[0].pulse.strength = {float(mono_cfg.opencarp_stimulus_strength):.6f}",
            ]
        )

        par_path = sim_dir / "sample.par"
        par_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return par_path

    def run_sample(self, context: SimulationContext) -> SimulationResult:
        if os.name == "nt":
            raise RuntimeError(
                "The openCARP production backend is Linux-only. "
                "Use the Windows smoke backend on this machine."
            )
        if not self.is_available():
            raise RuntimeError(
                f"{self.binary_name} not found on PATH; cannot run openCARP backend."
            )

        with tempfile.TemporaryDirectory(prefix="bpc_fno_opencarp_") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            mesh_base = tmp_dir / "mesh" / "slab"
            mesh_base.parent.mkdir(parents=True, exist_ok=True)
            sim_dir = tmp_dir / "sim"
            sim_dir.mkdir(parents=True, exist_ok=True)

            mesh_info = _write_regular_hexa_mesh(
                mesh_base=mesh_base,
                voxel_size_cm=context.slab.voxel_size_cm,
                fiber=context.fiber,
                cell_type_map=context.cell_type_map,
                fibrosis_mask=context.fibrosis_mask,
            )
            par_path = self._build_parameter_file(
                context=context,
                mesh_base=mesh_base,
                sim_dir=sim_dir,
            )

            command = [self.binary_name, "+F", str(par_path)]
            logger.info("Running openCARP sample via %s", " ".join(command))
            completed = subprocess.run(
                command,
                cwd=tmp_dir,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"{self.binary_name} failed with code {completed.returncode}.\n"
                    f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
                )

            vm_path = sim_dir / "vm.igb"
            if not vm_path.exists():
                raise FileNotFoundError(
                    f"openCARP completed but {vm_path} was not produced."
                )

            vm_nodes, _ = _parse_igb(vm_path)
            vm_voxels = _voltage_nodes_to_voxels(vm_nodes, context.slab.grid_shape)
            t_ms = np.asarray(context.output_times_ms, dtype=np.float32)
            if vm_voxels.shape[0] != t_ms.shape[0]:
                src_t = np.linspace(
                    0.0,
                    float(context.config.monodomain.total_time_ms),
                    vm_voxels.shape[0],
                    endpoint=False,
                    dtype=np.float64,
                )
                resampled = []
                for i in range(context.slab.grid_shape[0]):
                    for j in range(context.slab.grid_shape[1]):
                        for k in range(context.slab.grid_shape[2]):
                            resampled.append(
                                np.interp(t_ms, src_t, vm_voxels[:, i, j, k])
                            )
                vm_voxels = np.asarray(resampled, dtype=np.float32).T.reshape(
                    t_ms.shape[0], *context.slab.grid_shape
                )

            D_tensor = context.conductivity.get_tensor_field()
            J_i = []
            for V_field in vm_voxels:
                edge_order = 2 if min(context.slab.grid_shape) >= 3 else 1
                grad_V = np.stack(
                    np.gradient(
                        V_field, context.slab.voxel_size_cm, edge_order=edge_order
                    ),
                    axis=-1,
                )
                J_i.append(
                    -np.einsum("...ab,...b->...a", D_tensor, grad_V).astype(np.float32)
                )

            stim_p0_um, stim_p1_um = _box_from_pacing_site(
                context.pacing_site_voxel,
                context.slab.voxel_size_cm,
                float(context.config.monodomain.stimulus_radius_cm),
            )
            stimulus_mask = _stimulus_mask_from_box(
                context.slab.grid_shape,
                context.slab.voxel_size_cm,
                stim_p0_um,
                stim_p1_um,
            )

            activation_times_ms = _compute_activation_times(
                vm_voxels=np.asarray(vm_voxels, dtype=np.float32),
                t_ms=t_ms,
                threshold_mV=float(context.config.monodomain.activation_threshold_mV),
            ).astype(np.float32)

            metadata = {
                "simulation_method": "openCARP_monodomain",
                "monodomain_exactness": "fully_coupled",
                "smoke_only": False,
                "binary": self.binary_name,
                "mesh_files": {
                    "pts": str(mesh_info["pts_path"]),
                    "elem": str(mesh_info["elem_path"]),
                    "lon": str(mesh_info["lon_path"]),
                },
            }
            return SimulationResult(
                J_i=np.stack(J_i, axis=0),
                V_m=np.asarray(vm_voxels, dtype=np.float32)
                if context.save_vm
                else None,
                t_ms=t_ms,
                activation_times_ms=activation_times_ms,
                stimulus_mask=stimulus_mask,
                metadata=metadata,
            )


class OpenCARPAdapter(_BaseCarpAdapter):
    """Adapter for the ``openCARP`` binary."""

    def __init__(self) -> None:
        super().__init__(binary_name="openCARP")


class CarpCLIAdapter(_BaseCarpAdapter):
    """Adapter for the ``carp.pt`` binary."""

    def __init__(self) -> None:
        super().__init__(binary_name="carp.pt")


class OpenCARPBackend(SimulationBackend):
    """Production Linux backend using openCARP or carp.pt."""

    name = "opencarp"
    smoke_only = False
    requires_lookup_ionics = False
    coupling_description = "fully_coupled_opencarp_monodomain"

    def __init__(self) -> None:
        self._adapters = [OpenCARPAdapter(), CarpCLIAdapter()]

    def _resolve_adapter(self) -> _BaseCarpAdapter:
        for adapter in self._adapters:
            if adapter.is_available():
                return adapter
        raise RuntimeError(
            "Neither 'openCARP' nor 'carp.pt' was found on PATH."
        )

    def simulate(
        self,
        context: SimulationContext,
        ap_waveforms=None,
    ) -> SimulationResult:
        _ = ap_waveforms
        adapter = self._resolve_adapter()
        return adapter.run_sample(context)
