"""Fresh-interpreter worker for one reference-validation FEM solve."""

from __future__ import annotations

import pickle
import sys

from .forward import run_forward_model
from .reference_validation import ForwardFailure, forward_observation_from_result


def main() -> int:
    """Read one pickled solve request and return a pickled outcome."""

    displacements_m, config = pickle.loads(sys.stdin.buffer.read())
    try:
        result = run_forward_model(displacements_m, config)
        outcome = forward_observation_from_result(result)
    except Exception as error:  # the parent report must preserve failed rows
        outcome = ForwardFailure(type(error).__name__, str(error))
    sys.stdout.buffer.write(pickle.dumps(outcome))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
