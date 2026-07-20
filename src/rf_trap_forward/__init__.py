"""Public API for the RF-trap forward model and convergence validation."""

from .config import (
    ForwardModelConfig,
    GeometryConfig,
    MeshConfig,
    MinimaSearchConfig,
    SolverConfig,
)
from .forward import ForwardModelResult, run_forward_model
from .minima import LocalMinimum, MinimaSearchError
from .validation import (
    ConvergenceOutputPaths,
    ConvergenceReport,
    ConvergenceRunRecord,
    ConvergenceStudyConfig,
    CoordinateComparison,
    build_convergence_report,
    compare_successive_minima,
    run_convergence_study,
    write_convergence_outputs,
)

__all__ = [
    "ConvergenceOutputPaths",
    "ConvergenceReport",
    "ConvergenceRunRecord",
    "ConvergenceStudyConfig",
    "CoordinateComparison",
    "ForwardModelConfig",
    "ForwardModelResult",
    "GeometryConfig",
    "LocalMinimum",
    "MeshConfig",
    "MinimaSearchConfig",
    "MinimaSearchError",
    "SolverConfig",
    "build_convergence_report",
    "compare_successive_minima",
    "run_forward_model",
    "run_convergence_study",
    "write_convergence_outputs",
]
