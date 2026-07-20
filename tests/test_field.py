"""Focused tests for the gradient-recovery numerical algorithm."""

from __future__ import annotations

import numpy as np
from skfem import MeshTri

from rf_trap_forward.field import recover_nodal_electric_field


def test_linear_potential_has_exact_recovered_field() -> None:
    """Area-weighted recovery must exactly reproduce a globally linear gradient."""

    mesh = MeshTri.init_symmetric().refined(3)
    potential = 1.5 * mesh.p[0] - 2.0 * mesh.p[1] + 0.7
    recovered = recover_nodal_electric_field(mesh, potential)
    expected = np.repeat(np.asarray([[-1.5], [2.0]]), mesh.p.shape[1], axis=1)
    np.testing.assert_allclose(recovered, expected, rtol=1.0e-13, atol=1.0e-13)

