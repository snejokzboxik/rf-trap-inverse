"""End-to-end orchestration for one forward-model input."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import ForwardModelConfig
from .field import RecoveredField, recover_field
from .geometry import (
    TrapGeometry,
    build_geometry,
    build_geometry_from_absolute_displacements,
)
from .mesh import TrapMesh, generate_mesh
from .minima import LocalMinimum, MinimaDiagnostics, find_local_minima
from .minima_modes import (
    MinimaMode,
    MinimaModeResult,
    RobustMinimaConfig,
    run_minima_mode,
)
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
    minima_mode: MinimaMode = "recovered-gradient"
    minima_mode_result: MinimaModeResult | None = None

    def minima_positions_m(self) -> NDArray[np.float64]:
        """Return the angle-sorted minimum coordinates as an ``(n, 2)`` array."""

        if not self.minima:
            return np.empty((0, 2), dtype=float)
        return np.vstack([minimum.position_m for minimum in self.minima])


def run_forward_model(
    displacements_m: ArrayLike,
    config: ForwardModelConfig,
    *,
    minima_mode: MinimaMode = "recovered-gradient",
    robust_minima_config: RobustMinimaConfig | None = None,
) -> ForwardModelResult:
    """Run the FEM pipeline with an explicit, default-preserving minima mode."""

    geometry = build_geometry(config.geometry, displacements_m)
    return _run_forward_geometry(
        geometry,
        config,
        minima_mode,
        robust_minima_config,
    )


def run_forward_model_from_absolute_displacements(
    absolute_displacements_m: ArrayLike,
    config: ForwardModelConfig,
    *,
    minima_mode: MinimaMode = "recovered-gradient",
    robust_minima_config: RobustMinimaConfig | None = None,
) -> ForwardModelResult:
    """Run the FEM pipeline while moving all four electrodes in the lab frame."""

    geometry = build_geometry_from_absolute_displacements(
        config.geometry,
        absolute_displacements_m,
    )
    return _run_forward_geometry(
        geometry,
        config,
        minima_mode,
        robust_minima_config,
    )


def _run_forward_geometry(
    geometry: TrapGeometry,
    config: ForwardModelConfig,
    minima_mode: MinimaMode,
    robust_minima_config: RobustMinimaConfig | None,
) -> ForwardModelResult:
    """Run the shared solver and postprocessor for an already-built geometry."""

    trap_mesh = generate_mesh(geometry, config.mesh)
    solution = solve_potential(geometry, trap_mesh, config.solver)
    recovered_field = recover_field(solution)
    mode_result = None
    if minima_mode == "recovered-gradient":
        minima, diagnostics = find_local_minima(recovered_field, config.minima)
    else:
        mode_result = run_minima_mode(
            recovered_field,
            solution.potential_v,
            config.minima,
            mode=minima_mode,
            robust_config=robust_minima_config,
        )
        minima = mode_result.minima
        diagnostics = mode_result.legacy_diagnostics or MinimaDiagnostics(
            valid_coarse_points=0,
            coarse_candidates=0,
            refined_candidates=0,
            unique_candidates=len(mode_result.candidates),
            hessian_validated_candidates=0,
        )
    return ForwardModelResult(
        geometry=geometry,
        trap_mesh=trap_mesh,
        fem_solution=solution,
        recovered_field=recovered_field,
        minima=minima,
        minima_diagnostics=diagnostics,
        minima_mode=minima_mode,
        minima_mode_result=mode_result,
    )
