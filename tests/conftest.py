"""Shared deterministic configurations and numerical products for tests."""

from __future__ import annotations

import numpy as np
import pytest

from rf_trap_forward import (
    ForwardModelConfig,
    ForwardModelResult,
    run_forward_model,
)
from rf_trap_forward.demo import demonstrator_config, demonstrator_displacements_m
from rf_trap_forward.geometry import TrapGeometry, build_geometry
from rf_trap_forward.mesh import TrapMesh, generate_mesh
from rf_trap_forward.solver import FEMSolution, solve_potential


@pytest.fixture(scope="session")
def model_config() -> ForwardModelConfig:
    """Return the single provisional configuration used by integration tests."""

    return demonstrator_config()


@pytest.fixture(scope="session")
def displacements_m() -> np.ndarray:
    """Return one deterministic asymmetric displacement input in metres."""

    return demonstrator_displacements_m()


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
