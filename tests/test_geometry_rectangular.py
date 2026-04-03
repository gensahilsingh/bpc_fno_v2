from __future__ import annotations

import numpy as np

from bpc_fno.simulation.tissue.geometry import VentricularSlab


def test_rectangular_slab_shapes() -> None:
    slab = VentricularSlab(grid_size=(5, 4, 3), voxel_size_cm=0.1)

    assert slab.grid_shape == (5, 4, 3)
    assert slab.get_sdf().shape == (5, 4, 3)
    assert slab.get_fiber_field().shape == (5, 4, 3, 3)
    assert slab.get_cell_type_map().shape == (5, 4, 3)


def test_rectangular_slab_fibers_are_unit_vectors() -> None:
    slab = VentricularSlab(grid_size=(6, 3, 2), voxel_size_cm=0.1)
    fiber = slab.get_fiber_field()
    norms = np.linalg.norm(fiber, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-7)
