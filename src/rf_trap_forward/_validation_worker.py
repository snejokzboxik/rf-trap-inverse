"""Fresh-interpreter worker used to isolate each Gmsh convergence run."""

from __future__ import annotations

import pickle
import sys
from contextlib import redirect_stdout

from .validation import _run_isolated_case


def main() -> int:
    """Read one internal run payload and emit one pickled convergence record."""

    displacement_m, base_config, mesh_size_m, outer_radius_m = pickle.load(
        sys.stdin.buffer
    )
    with redirect_stdout(sys.stderr):
        record = _run_isolated_case(
            displacement_m,
            base_config,
            mesh_size_m,
            outer_radius_m,
        )
    pickle.dump(record, sys.stdout.buffer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
