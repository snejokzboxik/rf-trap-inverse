"""Fresh-process robust-only FEM worker for Milestone 9 calibration."""

from __future__ import annotations

import pickle
import sys
import time

import numpy as np

from .field import recover_field
from .geometry import build_geometry, build_geometry_from_absolute_displacements
from .mesh import generate_mesh
from .minima_modes import RobustMinimaConfig, run_minima_mode
from .solver import solve_potential


def main() -> int:
    """Run one robust-only forward case and return serializable diagnostics."""

    started = time.perf_counter()
    try:
        displacements_m, config, robust_config = pickle.loads(sys.stdin.buffer.read())
        if not isinstance(robust_config, RobustMinimaConfig):
            raise TypeError("worker requires RobustMinimaConfig")
        if np.asarray(displacements_m).shape in ((4, 2), (8,)):
            geometry = build_geometry_from_absolute_displacements(
                config.geometry,
                displacements_m,
            )
        else:
            geometry = build_geometry(config.geometry, displacements_m)
        trap_mesh = generate_mesh(geometry, config.mesh)
        solution = solve_potential(geometry, trap_mesh, config.solver)
        recovered = recover_field(solution)
        result = run_minima_mode(
            recovered,
            solution.potential_v,
            config.minima,
            mode="robust",
            robust_config=robust_config,
        )
        outcome = {
            "ok": True,
            "modes": {"robust": result},
            "mode_errors": {},
            "node_count": trap_mesh.number_of_nodes,
            "triangle_count": trap_mesh.number_of_triangles,
            "relative_free_residual": solution.relative_free_residual,
            "runtime_seconds": time.perf_counter() - started,
        }
    except Exception as error:
        outcome = {
            "ok": False,
            "runtime_seconds": time.perf_counter() - started,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
    sys.stdout.buffer.write(pickle.dumps(outcome))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
