"""Fresh-process one-electrode basis-field evaluator for Milestone 6."""

from __future__ import annotations

import pickle
import sys
from dataclasses import replace

import numpy as np

from .field import recover_field
from .geometry import build_geometry
from .mesh import generate_mesh
from .solver import solve_potential


def main() -> int:
    """Read one geometry request and return four one-hot recovered fields."""

    displacements_m, config, target_positions_m = pickle.loads(
        sys.stdin.buffer.read()
    )
    try:
        geometry = build_geometry(config.geometry, displacements_m)
        trap_mesh = generate_mesh(geometry, config.mesh)
        basis_fields = []
        for electrode in range(4):
            potentials = tuple(
                1.0 if index == electrode else 0.0 for index in range(4)
            )
            basis_geometry = replace(
                geometry,
                config=replace(
                    geometry.config,
                    electrode_potentials_v=potentials,
                ),
            )
            solution = solve_potential(basis_geometry, trap_mesh, config.solver)
            basis_fields.append(recover_field(solution).evaluate(target_positions_m))
        outcome: dict[str, object] = {
            "ok": True,
            "basis_fields_v_per_m": np.stack(basis_fields, axis=2),
            "node_count": trap_mesh.number_of_nodes,
            "triangle_count": trap_mesh.number_of_triangles,
        }
    except Exception as error:
        outcome = {
            "ok": False,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
    sys.stdout.buffer.write(pickle.dumps(outcome))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
