"""Public API for the RF-trap forward model and convergence validation."""

from .config import (
    ForwardModelConfig,
    GeometryConfig,
    MeshConfig,
    MeshSizeFieldConfig,
    MinimaSearchConfig,
    SolverConfig,
)
from .forward import (
    ForwardModelResult,
    run_forward_model,
    run_forward_model_from_absolute_displacements,
)
from .geometry import (
    absolute_displacements_m,
    build_geometry_from_absolute_displacements,
)
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
    "MeshSizeFieldConfig",
    "MinimaSearchConfig",
    "MinimaSearchError",
    "SolverConfig",
    "build_convergence_report",
    "absolute_displacements_m",
    "build_geometry_from_absolute_displacements",
    "compare_successive_minima",
    "run_forward_model",
    "run_forward_model_from_absolute_displacements",
    "run_convergence_study",
    "write_convergence_outputs",
]
