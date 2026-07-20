"""Shared deterministic configurations and numerical products for tests."""

from __future__ import annotations

import numpy as np
import pytest

from rf_trap_forward import (
    ForwardModelConfig,
    GeometryConfig,
    MeshConfig,
    MinimaSearchConfig,
    SolverConfig,
    ForwardModelResult,
    run_forward_model,
)
from rf_trap_forward.geometry import TrapGeometry, build_geometry
from rf_trap_forward.mesh import TrapMesh, generate_mesh
from rf_trap_forward.solver import FEMSolution, solve_potential


@pytest.fixture(scope="session")
def model_config() -> ForwardModelConfig:
    """Return the single provisional configuration used by integration tests."""

    return ForwardModelConfig(
        geometry=GeometryConfig(
            electrode_radius_m=0.32e-3,
            nominal_centers_m=(
                (1.10e-3, 0.0),
                (0.0, 1.10e-3),
                (-1.10e-3, 0.0),
                (0.0, -1.10e-3),
            ),
            outer_radius_m=4.0e-3,
        ),
        mesh=MeshConfig(characteristic_length_m=0.08e-3),
        solver=SolverConfig(),
        minima=MinimaSearchConfig(
            search_half_extent_m=0.70e-3,
            coarse_grid_points_per_axis=71,
            merge_distance_m=0.02e-3,
            hessian_step_m=0.004e-3,
        ),
    )


@pytest.fixture(scope="session")
def displacements_m() -> np.ndarray:
    """Return one deterministic asymmetric displacement input in metres."""

    return np.asarray([120.0, -80.0, -150.0, 110.0, 90.0, 160.0]) * 1.0e-6


@pytest.fixture(scope="session")
def geometry(model_config: ForwardModelConfig, displacements_m: np.ndarray) -> TrapGeometry:
    """Build the shared displaced geometry."""

    return build_geometry(model_config.geometry, displacements_m)


@pytest.fixture(scope="session")
def trap_mesh(geometry: TrapGeometry, model_config: ForwardModelConfig) -> TrapMesh:
    """Generate the shared conforming mesh once."""

    return generate_mesh(geometry, model_config.mesh)


@pytest.fixture(scope="session")
def fem_solution(
    geometry: TrapGeometry,
    trap_mesh: TrapMesh,
    model_config: ForwardModelConfig,
) -> FEMSolution:
    """Solve the shared FEM system once."""

    return solve_potential(geometry, trap_mesh, model_config.solver)


@pytest.fixture(scope="session")
def forward_result(
    model_config: ForwardModelConfig,
    displacements_m: np.ndarray,
) -> ForwardModelResult:
    """Run the complete milestone-one pipeline once."""

    return run_forward_model(displacements_m, model_config)
