"""Fresh-process worker for recovered-gradient candidate diagnostics."""

from __future__ import annotations

import pickle
import sys

from .forward import run_forward_model
from .numerical_audit import diagnose_recovered_minima


def main() -> int:
    """Read one audit case from stdin and return a small pickled result."""

    try:
        displacements_m, config, artifact_action = pickle.loads(sys.stdin.buffer.read())
        result = run_forward_model(displacements_m, config)
        records = diagnose_recovered_minima(
            result,
            artifact_action=artifact_action,
        )
        outcome = {
            "ok": True,
            "node_count": result.trap_mesh.number_of_nodes,
            "triangle_count": result.trap_mesh.number_of_triangles,
            "relative_free_residual": result.fem_solution.relative_free_residual,
            "records": records,
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
