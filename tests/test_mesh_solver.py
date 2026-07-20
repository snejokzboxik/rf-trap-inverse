"""Integration tests for meshing and the Laplace solve."""

from __future__ import annotations

import numpy as np

from rf_trap_forward import ForwardModelConfig
from rf_trap_forward.geometry import TrapGeometry
from rf_trap_forward.mesh import TrapMesh, generate_mesh


def test_mesh_is_nonempty_and_conforms_to_vacuum(geometry, trap_mesh) -> None:
    """All element centroids must lie outside electrodes and inside the outer disk."""

    assert trap_mesh.number_of_nodes > 100
    assert trap_mesh.number_of_triangles > 100
    centroids = np.mean(trap_mesh.mesh.p[:, trap_mesh.mesh.t], axis=1).T
    assert np.all(geometry.contains_points(centroids))
    classified = np.union1d(
        trap_mesh.outer_boundary_nodes,
        trap_mesh.electrode_boundary_nodes,
    )
    np.testing.assert_array_equal(np.sort(classified), np.sort(trap_mesh.mesh.boundary_nodes()))


def test_fem_solution_satisfies_boundary_values_and_maximum_principle(fem_solution) -> None:
    """The solved potential must meet Dirichlet data and remain between them."""

    solution = fem_solution
    mesh = solution.trap_mesh
    np.testing.assert_allclose(
        solution.potential_v[mesh.electrode_boundary_nodes],
        solution.geometry.config.electrode_potential_v,
        atol=0.0,
    )
    np.testing.assert_allclose(
        solution.potential_v[mesh.outer_boundary_nodes],
        solution.geometry.config.outer_potential_v,
        atol=0.0,
    )
    assert np.min(solution.potential_v) >= 0.0
    assert np.max(solution.potential_v) <= 1.0
    assert solution.relative_free_residual < 1.0e-10


def test_fixed_seed_reproduces_the_mesh(
    geometry: TrapGeometry,
    trap_mesh: TrapMesh,
    model_config: ForwardModelConfig,
) -> None:
    """The configured one-thread Gmsh run must reproduce nodes and connectivity."""

    repeated = generate_mesh(geometry, model_config.mesh)
    np.testing.assert_array_equal(repeated.mesh.p, trap_mesh.mesh.p)
    np.testing.assert_array_equal(repeated.mesh.t, trap_mesh.mesh.t)
