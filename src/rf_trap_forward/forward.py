"""End-to-end orchestration for one forward-model input."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import ForwardModelConfig
from .field import RecoveredField, recover_field
from .geometry import TrapGeometry, build_geometry
from .mesh import TrapMesh, generate_mesh
from .minima import LocalMinimum, MinimaDiagnostics, find_local_minima
from .solver import FEMSolution, solve_potential


@dataclass(frozen=True)
class ForwardModelResult:
    """Numerical products and diagnostics for one displacement configuration."""

    geometry: TrapGeometry
    trap_mesh: TrapMesh
    fem_solution: FEMSolution
    recovered_field: RecoveredField
    minima: tuple[LocalMinimum, ...]
    minima_diagnostics: MinimaDiagnostics

    def minima_positions_m(self) -> NDArray[np.float64]:
        """Return the angle-sorted minimum coordinates as an ``(n, 2)`` array."""

        return np.vstack([minimum.position_m for minimum in self.minima])


def run_forward_model(
    displacements_m: ArrayLike,
    config: ForwardModelConfig,
) -> ForwardModelResult:
    """Run geometry, meshing, FEM, field recovery, and minima search once."""

    geometry = build_geometry(config.geometry, displacements_m)
    trap_mesh = generate_mesh(geometry, config.mesh)
    solution = solve_potential(geometry, trap_mesh, config.solver)
    recovered_field = recover_field(solution)
    minima, diagnostics = find_local_minima(recovered_field, config.minima)
    return ForwardModelResult(
        geometry=geometry,
        trap_mesh=trap_mesh,
        fem_solution=solution,
        recovered_field=recovered_field,
        minima=minima,
        minima_diagnostics=diagnostics,
    )
