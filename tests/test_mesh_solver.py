"""Integration tests for meshing and the Laplace solve."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from rf_trap_forward import ForwardModelConfig
from rf_trap_forward.geometry import TrapGeometry
from rf_trap_forward.geometry import build_geometry
from rf_trap_forward.mesh import (
    TrapMesh,
    estimate_central_triangle_count,
    generate_mesh,
)
from rf_trap_forward.real_scale import locally_refined_real_scale_forward_config
from rf_trap_forward.solver import solve_potential


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
    """Mesh output must be independent of an intervening different geometry."""

    alternate_config = replace(
        model_config.geometry,
        outer_radius_m=3.5e-3,
    )
    alternate_geometry = build_geometry(
        alternate_config,
        geometry.displacements_m,
    )
    generate_mesh(alternate_geometry, model_config.mesh)
    repeated = generate_mesh(geometry, model_config.mesh)
    np.testing.assert_array_equal(repeated.mesh.p, trap_mesh.mesh.p)
    np.testing.assert_array_equal(repeated.mesh.t, trap_mesh.mesh.t)


def test_solver_supports_explicit_per_electrode_dirichlet_values(
    geometry: TrapGeometry,
    trap_mesh: TrapMesh,
    model_config: ForwardModelConfig,
) -> None:
    """A diagnostic alternating-polarity solve must impose each boundary value."""

    potentials = (1.0, -1.0, -1.0, 1.0)
    alternating_geometry = TrapGeometry(
        config=replace(
            geometry.config,
            electrode_potentials_v=potentials,
        ),
        centers_m=geometry.centers_m,
        displacements_m=geometry.displacements_m,
    )
    solution = solve_potential(
        alternating_geometry,
        trap_mesh,
        model_config.solver,
    )
    for nodes, expected in zip(
        trap_mesh.electrode_boundary_nodes_by_electrode,
        potentials,
        strict=True,
    ):
        np.testing.assert_allclose(solution.potential_v[nodes], expected, atol=0.0)
    assert np.min(solution.potential_v) >= -1.0
    assert np.max(solution.potential_v) <= 1.0


def test_central_triangle_estimate_is_monotone_and_validated() -> None:
    """The preflight estimate must grow quadratically as central h shrinks."""

    coarse = estimate_central_triangle_count(8.0e-3, 0.50e-3)
    fine = estimate_central_triangle_count(8.0e-3, 0.25e-3)
    assert fine == 4 * coarse or abs(fine - 4 * coarse) <= 3
    with np.testing.assert_raises(ValueError):
        estimate_central_triangle_count(8.0e-3, 0.0)


def test_local_size_field_refines_the_central_region() -> None:
    """An actual Gmsh mesh must be finer centrally while retaining all markers."""

    config = locally_refined_real_scale_forward_config(
        central_mesh_size_m=0.50e-3,
    )
    geometry = build_geometry(config.geometry, np.zeros(6))
    mesh = generate_mesh(geometry, config.mesh)
    points = mesh.mesh.p.T
    triangles = mesh.mesh.t.T
    edge_pairs = np.vstack(
        (triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)])
    )
    edges = np.unique(np.sort(edge_pairs, axis=1), axis=0)
    edge_vectors = points[edges[:, 1]] - points[edges[:, 0]]
    lengths = np.linalg.norm(edge_vectors, axis=1)
    midpoint_radii = np.linalg.norm(
        0.5 * (points[edges[:, 1]] + points[edges[:, 0]]),
        axis=1,
    )
    central_lengths = lengths[midpoint_radii <= 6.0e-3]
    outer_lengths = lengths[midpoint_radii >= 35.0e-3]
    assert central_lengths.size > 100
    assert outer_lengths.size > 100
    assert np.median(central_lengths) < 0.40 * np.median(outer_lengths)
    classified = np.union1d(mesh.outer_boundary_nodes, mesh.electrode_boundary_nodes)
    np.testing.assert_array_equal(
        np.sort(classified),
        np.sort(mesh.mesh.boundary_nodes()),
    )
