"""Deterministic synthetic dataset generation from the validated FEM path."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .absolute_validation import wolfram_to_fem_absolute_displacements_m
from .calibrated_validation import _run_minima_worker
from .config import ForwardModelConfig
from .dataset import sort_points_by_polar_angle
from .geometry import build_geometry_from_absolute_displacements
from .minima_modes import MinimaModeResult, RobustMinimaConfig
from .real_scale import (
    REAL_ELECTRODE_RADIUS_M,
    REAL_INNER_RADIUS_M,
    REAL_OUTER_BOUNDARY_RADIUS_M,
    locally_refined_real_scale_forward_config,
)

MeshMode = Literal["practical"]
PRACTICAL_CENTRAL_MESH_SIZE_M = 500.0e-6
DEFAULT_AMBIGUOUS_MINIMUM_DISTANCE_M = 0.15e-3

CLEAN_CSV_COLUMNS = (
    "sample_id",
    "seed",
    "w1_dx_m",
    "w1_dy_m",
    "w2_dx_m",
    "w2_dy_m",
    "w3_dx_m",
    "w3_dy_m",
    "w4_dx_m",
    "w4_dy_m",
    "f1_dx_m",
    "f1_dy_m",
    "f2_dx_m",
    "f2_dy_m",
    "f3_dx_m",
    "f3_dy_m",
    "f4_dx_m",
    "f4_dy_m",
    "min1_x_m",
    "min1_y_m",
    "min2_x_m",
    "min2_y_m",
    "min3_x_m",
    "min3_y_m",
    "min_pairwise_distance_m",
    "rejected_candidate_count",
    "status",
)

REJECTED_CSV_COLUMNS = CLEAN_CSV_COLUMNS + (
    "accepted_candidate_count",
    "total_candidate_count",
    "selected_interpolation_sensitive_count",
    "node_count",
    "triangle_count",
    "relative_free_residual",
    "runtime_seconds",
    "error_type",
    "error_message",
)


@dataclass(frozen=True)
class SyntheticDatasetConfig:
    """Sampling, mesh, ambiguity, concurrency, and large-run controls."""

    n: int = 100
    seed: int = 123
    max_displacement_m: float = 500.0e-6
    mesh_mode: MeshMode = "practical"
    batch_size: int = 3
    ambiguous_minimum_distance_m: float = DEFAULT_AMBIGUOUS_MINIMUM_DISTANCE_M
    allow_large_n: bool = False

    def __post_init__(self) -> None:
        """Validate generator controls without changing their stated units."""

        if self.n <= 0:
            raise ValueError("n must be positive")
        if self.n > 1000 and not self.allow_large_n:
            raise ValueError(
                "n > 1000 requires --allow-large-n because generation may take many hours."
            )
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not np.isfinite(self.max_displacement_m) or self.max_displacement_m <= 0.0:
            raise ValueError("max_displacement_m must be finite and positive")
        if self.mesh_mode != "practical":
            raise ValueError("mesh_mode must be 'practical'")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if (
            not np.isfinite(self.ambiguous_minimum_distance_m)
            or self.ambiguous_minimum_distance_m <= 0.0
        ):
            raise ValueError(
                "ambiguous_minimum_distance_m must be finite and positive"
            )


@dataclass(frozen=True)
class SyntheticSolveResult:
    """Serializable robust-FEM result needed for dataset classification."""

    minima_positions_m: NDArray[np.float64]
    accepted_candidate_count: int
    rejected_candidate_count: int
    total_candidate_count: int
    selected_interpolation_sensitive_count: int
    node_count: int
    triangle_count: int
    relative_free_residual: float
    runtime_seconds: float
    error_type: str = ""
    error_message: str = ""

    def __post_init__(self) -> None:
        """Copy and validate the variable-length minima array."""

        minima = np.asarray(self.minima_positions_m, dtype=float)
        if minima.ndim != 2 or minima.shape[1] != 2 or not np.all(np.isfinite(minima)):
            raise ValueError("minima_positions_m must have finite shape (n, 2)")
        counts = (
            self.accepted_candidate_count,
            self.rejected_candidate_count,
            self.total_candidate_count,
            self.selected_interpolation_sensitive_count,
            self.node_count,
            self.triangle_count,
        )
        if any(value < 0 for value in counts):
            raise ValueError("solver diagnostic counts must be non-negative")
        if not np.isfinite(self.runtime_seconds) or self.runtime_seconds < 0.0:
            raise ValueError("runtime_seconds must be finite and non-negative")
        object.__setattr__(self, "minima_positions_m", minima.copy())

    @classmethod
    def failure(
        cls,
        error_type: str,
        error_message: str,
        *,
        runtime_seconds: float = 0.0,
    ) -> SyntheticSolveResult:
        """Return a diagnostic failure without fabricated minima."""

        return cls(
            minima_positions_m=np.empty((0, 2), dtype=float),
            accepted_candidate_count=0,
            rejected_candidate_count=0,
            total_candidate_count=0,
            selected_interpolation_sensitive_count=0,
            node_count=0,
            triangle_count=0,
            relative_free_residual=float("nan"),
            runtime_seconds=runtime_seconds,
            error_type=error_type,
            error_message=error_message,
        )


@dataclass(frozen=True)
class SyntheticSampleRecord:
    """One sampled input, transformed FEM input, result, and split status."""

    sample_id: int
    seed: int
    wolfram_displacements_m: NDArray[np.float64]
    fem_displacements_m: NDArray[np.float64]
    minima_positions_m: NDArray[np.float64]
    min_pairwise_distance_m: float
    rejected_candidate_count: int
    status: str
    accepted_candidate_count: int = 0
    total_candidate_count: int = 0
    selected_interpolation_sensitive_count: int = 0
    node_count: int = 0
    triangle_count: int = 0
    relative_free_residual: float = float("nan")
    runtime_seconds: float = 0.0
    error_type: str = ""
    error_message: str = ""

    def __post_init__(self) -> None:
        """Validate record shapes and copy mutable numerical inputs."""

        wolfram = np.asarray(self.wolfram_displacements_m, dtype=float)
        fem = np.asarray(self.fem_displacements_m, dtype=float)
        minima = np.asarray(self.minima_positions_m, dtype=float)
        if wolfram.shape != (4, 2) or fem.shape != (4, 2):
            raise ValueError("displacement arrays must have shape (4, 2)")
        if minima.ndim != 2 or minima.shape[1] != 2 or minima.shape[0] > 3:
            raise ValueError("minima_positions_m must have shape (n<=3, 2)")
        if not np.all(np.isfinite(wolfram)) or not np.all(np.isfinite(fem)):
            raise ValueError("displacements must be finite")
        if not np.all(np.isfinite(minima)):
            raise ValueError("available minima must be finite")
        object.__setattr__(self, "wolfram_displacements_m", wolfram.copy())
        object.__setattr__(self, "fem_displacements_m", fem.copy())
        object.__setattr__(self, "minima_positions_m", minima.copy())


@dataclass(frozen=True)
class SyntheticDatasetResult:
    """Complete requested sample set and aggregate wall-clock runtime."""

    config: SyntheticDatasetConfig
    records: tuple[SyntheticSampleRecord, ...]
    runtime_seconds: float

    @property
    def clean_records(self) -> tuple[SyntheticSampleRecord, ...]:
        """Return records admitted to the training split."""

        return tuple(item for item in self.records if item.status == "clean")

    @property
    def rejected_records(self) -> tuple[SyntheticSampleRecord, ...]:
        """Return all failures and ambiguous or non-three topologies."""

        return tuple(item for item in self.records if item.status != "clean")


@dataclass(frozen=True)
class SyntheticDatasetOutputPaths:
    """Requested generated-dataset artifact locations."""

    clean_csv: Path
    rejected_csv: Path
    summary_json: Path
    readme_markdown: Path


SyntheticWorker = Callable[
    [NDArray[np.float64], ForwardModelConfig, RobustMinimaConfig],
    SyntheticSolveResult,
]
ProgressCallback = Callable[[int, int, float], None]


def sample_wolfram_displacements_m(
    n: int,
    seed: int,
    max_displacement_m: float,
) -> NDArray[np.float64]:
    """Sample deterministic uniform ``(n, 4, 2)`` Wolfram-order inputs.

    The caller owns the large-run policy.  This helper deliberately does not
    recreate ``SyntheticDatasetConfig``, which would otherwise lose its
    validated ``allow_large_n`` acknowledgement.
    """

    if n <= 0:
        raise ValueError("n must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if not np.isfinite(max_displacement_m) or max_displacement_m <= 0.0:
        raise ValueError("max_displacement_m must be finite and positive")
    generator = np.random.default_rng(seed)
    return generator.uniform(
        -max_displacement_m,
        max_displacement_m,
        size=(n, 4, 2),
    )


def minimum_pairwise_distance_m(points_m: ArrayLike) -> float:
    """Return the smallest Euclidean separation among at least two points."""

    points = np.asarray(points_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        raise ValueError("points_m must have shape (n>=2, 2)")
    if not np.all(np.isfinite(points)):
        raise ValueError("points_m must be finite")
    differences = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    distances = np.linalg.norm(differences, axis=2)
    return float(np.min(distances[np.triu_indices(points.shape[0], k=1)]))


def practical_generator_forward_config(mesh_mode: MeshMode) -> ForwardModelConfig:
    """Return the named 500 µm real-scale dataset-generation configuration."""

    if mesh_mode != "practical":
        raise ValueError("mesh_mode must be 'practical'")
    return locally_refined_real_scale_forward_config(
        central_mesh_size_m=PRACTICAL_CENTRAL_MESH_SIZE_M,
    )


def generate_synthetic_dataset(
    config: SyntheticDatasetConfig,
    *,
    worker: SyntheticWorker | None = None,
    progress_callback: ProgressCallback | None = None,
) -> SyntheticDatasetResult:
    """Sample inputs, solve valid geometries, and split clean from rejected rows.

    ``batch_size`` is the maximum number of concurrent fresh FEM subprocesses.
    Returned records are always sorted by deterministic one-based sample ID.
    """

    started = time.perf_counter()
    forward_config = practical_generator_forward_config(config.mesh_mode)
    robust_config = RobustMinimaConfig()
    selected_worker = worker or _default_fem_worker
    sampled = sample_wolfram_displacements_m(
        config.n,
        config.seed,
        config.max_displacement_m,
    )
    prepared = []
    records: dict[int, SyntheticSampleRecord] = {}
    for sample_id, raw in enumerate(sampled, start=1):
        transformed = wolfram_to_fem_absolute_displacements_m(raw)
        try:
            build_geometry_from_absolute_displacements(
                forward_config.geometry,
                transformed,
            )
        except ValueError as error:
            records[sample_id] = _record_from_failure(
                sample_id,
                config.seed,
                raw,
                transformed,
                "geometry_overlap",
                type(error).__name__,
                str(error),
            )
        else:
            prepared.append((sample_id, raw.copy(), transformed))
    completed = len(records)
    if progress_callback is not None and completed:
        progress_callback(completed, config.n, time.perf_counter() - started)
    with ThreadPoolExecutor(
        max_workers=min(config.batch_size, max(1, len(prepared)))
    ) as executor:
        futures = {
            executor.submit(
                _safe_worker_call,
                selected_worker,
                transformed,
                forward_config,
                robust_config,
            ): (sample_id, raw, transformed)
            for sample_id, raw, transformed in prepared
        }
        for future in as_completed(futures):
            sample_id, raw, transformed = futures[future]
            solve = future.result()
            records[sample_id] = _record_from_solve(
                sample_id,
                config,
                raw,
                transformed,
                solve,
            )
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, config.n, time.perf_counter() - started)
    ordered = tuple(records[index] for index in range(1, config.n + 1))
    return SyntheticDatasetResult(
        config=config,
        records=ordered,
        runtime_seconds=time.perf_counter() - started,
    )


def write_synthetic_dataset(
    result: SyntheticDatasetResult,
    output_directory: str | Path,
) -> SyntheticDatasetOutputPaths:
    """Write stable clean/rejected CSVs, JSON summary, and dataset README."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = SyntheticDatasetOutputPaths(
        clean_csv=output / "synthetic_clean.csv",
        rejected_csv=output / "synthetic_rejected.csv",
        summary_json=output / "synthetic_summary.json",
        readme_markdown=output / "README.md",
    )
    _write_csv(
        paths.clean_csv,
        CLEAN_CSV_COLUMNS,
        [_csv_record(item, rejected=False) for item in result.clean_records],
    )
    _write_csv(
        paths.rejected_csv,
        REJECTED_CSV_COLUMNS,
        [_csv_record(item, rejected=True) for item in result.rejected_records],
    )
    summary = _summary_record(result)
    paths.summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    paths.readme_markdown.write_text(
        _dataset_readme(result, summary),
        encoding="utf-8",
    )
    return paths


def _default_fem_worker(
    fem_displacements_m: NDArray[np.float64],
    config: ForwardModelConfig,
    robust_config: RobustMinimaConfig,
) -> SyntheticSolveResult:
    outcome = _run_minima_worker(fem_displacements_m, config, robust_config)
    if not bool(outcome.get("ok")):
        return SyntheticSolveResult.failure(
            str(outcome.get("error_type", "WorkerError")),
            str(outcome.get("error_message", "unknown worker failure")),
            runtime_seconds=float(outcome.get("runtime_seconds", 0.0)),
        )
    modes = outcome.get("modes", {})
    robust = modes.get("robust") if isinstance(modes, dict) else None
    if not isinstance(robust, MinimaModeResult):
        return SyntheticSolveResult.failure(
            "RobustModeMissing",
            "fresh-process worker did not return robust minima",
            runtime_seconds=float(outcome.get("runtime_seconds", 0.0)),
        )
    positions = (
        np.vstack([item.position_m for item in robust.minima])
        if robust.minima
        else np.empty((0, 2), dtype=float)
    )
    return SyntheticSolveResult(
        minima_positions_m=positions,
        accepted_candidate_count=robust.accepted_candidates,
        rejected_candidate_count=robust.rejected_candidates,
        total_candidate_count=len(robust.candidates),
        selected_interpolation_sensitive_count=(
            robust.selected_interpolation_sensitive
        ),
        node_count=int(outcome["node_count"]),
        triangle_count=int(outcome["triangle_count"]),
        relative_free_residual=float(outcome["relative_free_residual"]),
        runtime_seconds=float(outcome["runtime_seconds"]),
    )


def _safe_worker_call(
    worker: SyntheticWorker,
    fem_displacements_m: NDArray[np.float64],
    config: ForwardModelConfig,
    robust_config: RobustMinimaConfig,
) -> SyntheticSolveResult:
    try:
        result = worker(fem_displacements_m, config, robust_config)
    except Exception as error:
        return SyntheticSolveResult.failure(type(error).__name__, str(error))
    if not isinstance(result, SyntheticSolveResult):
        return SyntheticSolveResult.failure(
            "WorkerProtocolError",
            "worker must return SyntheticSolveResult",
        )
    return result


def _record_from_solve(
    sample_id: int,
    config: SyntheticDatasetConfig,
    raw: NDArray[np.float64],
    transformed: NDArray[np.float64],
    solve: SyntheticSolveResult,
) -> SyntheticSampleRecord:
    positions = (
        sort_points_by_polar_angle(solve.minima_positions_m)
        if solve.minima_positions_m.size
        else np.empty((0, 2), dtype=float)
    )
    pairwise = (
        minimum_pairwise_distance_m(positions)
        if positions.shape[0] >= 2
        else float("nan")
    )
    if solve.error_type:
        status = "solver_failed"
    elif solve.accepted_candidate_count != 3 or positions.shape != (3, 2):
        status = "not_exactly_three_robust_minima"
    elif pairwise < config.ambiguous_minimum_distance_m:
        status = "ambiguous_branch"
    else:
        status = "clean"
    return SyntheticSampleRecord(
        sample_id=sample_id,
        seed=config.seed,
        wolfram_displacements_m=raw,
        fem_displacements_m=transformed,
        minima_positions_m=positions,
        min_pairwise_distance_m=pairwise,
        rejected_candidate_count=solve.rejected_candidate_count,
        status=status,
        accepted_candidate_count=solve.accepted_candidate_count,
        total_candidate_count=solve.total_candidate_count,
        selected_interpolation_sensitive_count=(
            solve.selected_interpolation_sensitive_count
        ),
        node_count=solve.node_count,
        triangle_count=solve.triangle_count,
        relative_free_residual=solve.relative_free_residual,
        runtime_seconds=solve.runtime_seconds,
        error_type=solve.error_type,
        error_message=solve.error_message,
    )


def _record_from_failure(
    sample_id: int,
    seed: int,
    raw: NDArray[np.float64],
    transformed: NDArray[np.float64],
    status: str,
    error_type: str,
    error_message: str,
) -> SyntheticSampleRecord:
    return SyntheticSampleRecord(
        sample_id=sample_id,
        seed=seed,
        wolfram_displacements_m=raw,
        fem_displacements_m=transformed,
        minima_positions_m=np.empty((0, 2), dtype=float),
        min_pairwise_distance_m=float("nan"),
        rejected_candidate_count=0,
        status=status,
        error_type=error_type,
        error_message=error_message,
    )


def _csv_record(
    record: SyntheticSampleRecord,
    *,
    rejected: bool,
) -> dict[str, object]:
    row: dict[str, object] = {
        "sample_id": record.sample_id,
        "seed": record.seed,
        "min_pairwise_distance_m": _finite_or_blank(
            record.min_pairwise_distance_m
        ),
        "rejected_candidate_count": record.rejected_candidate_count,
        "status": record.status,
    }
    for prefix, values in (
        ("w", record.wolfram_displacements_m),
        ("f", record.fem_displacements_m),
    ):
        for index, pair in enumerate(values, start=1):
            row[f"{prefix}{index}_dx_m"] = float(pair[0])
            row[f"{prefix}{index}_dy_m"] = float(pair[1])
    for index in range(3):
        if index < record.minima_positions_m.shape[0]:
            row[f"min{index + 1}_x_m"] = float(record.minima_positions_m[index, 0])
            row[f"min{index + 1}_y_m"] = float(record.minima_positions_m[index, 1])
        else:
            row[f"min{index + 1}_x_m"] = ""
            row[f"min{index + 1}_y_m"] = ""
    if rejected:
        row.update(
            {
                "accepted_candidate_count": record.accepted_candidate_count,
                "total_candidate_count": record.total_candidate_count,
                "selected_interpolation_sensitive_count": (
                    record.selected_interpolation_sensitive_count
                ),
                "node_count": record.node_count,
                "triangle_count": record.triangle_count,
                "relative_free_residual": _finite_or_blank(
                    record.relative_free_residual
                ),
                "runtime_seconds": record.runtime_seconds,
                "error_type": record.error_type,
                "error_message": record.error_message,
            }
        )
    columns = REJECTED_CSV_COLUMNS if rejected else CLEAN_CSV_COLUMNS
    return {column: row.get(column, "") for column in columns}


def _finite_or_blank(value: float) -> float | str:
    return float(value) if np.isfinite(value) else ""


def _write_csv(
    path: Path,
    columns: Sequence[str],
    records: Sequence[dict[str, object]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(records)


def _summary_record(result: SyntheticDatasetResult) -> dict[str, object]:
    statuses = Counter(item.status for item in result.records)
    solver_runtime = sum(item.runtime_seconds for item in result.records)
    clean_rejected_candidates = [
        item.rejected_candidate_count for item in result.clean_records
    ]
    return {
        "requested_samples": result.config.n,
        "completed_samples": len(result.records),
        "clean_samples": len(result.clean_records),
        "rejected_samples": len(result.rejected_records),
        "ambiguous_branch_count": statuses.get("ambiguous_branch", 0),
        "clean_rows_with_rejected_candidates": sum(
            value > 0 for value in clean_rejected_candidates
        ),
        "clean_rejected_candidate_total": sum(clean_rejected_candidates),
        "clean_rejected_candidate_maximum": max(
            clean_rejected_candidates,
            default=0,
        ),
        "status_counts": dict(sorted(statuses.items())),
        "seed": result.config.seed,
        "max_displacement_m": result.config.max_displacement_m,
        "max_displacement_um": 1.0e6 * result.config.max_displacement_m,
        "mesh_mode": result.config.mesh_mode,
        "central_mesh_size_m": PRACTICAL_CENTRAL_MESH_SIZE_M,
        "batch_size": result.config.batch_size,
        "ambiguous_minimum_distance_m": (
            result.config.ambiguous_minimum_distance_m
        ),
        "wall_runtime_seconds": result.runtime_seconds,
        "summed_worker_runtime_seconds": solver_runtime,
        "coordinate_units": "metres",
        "raw_electrode_order": ["W1 upper-right", "W2 lower-right", "W3 upper-left", "W4 lower-left"],
        "fem_electrode_order": ["F1 upper-left", "F2 upper-right", "F3 lower-left", "F4 lower-right"],
        "wolfram_to_fem_transform": "[-W3, -W1, -W4, -W2]",
        "outer_boundary_radius_m": REAL_OUTER_BOUNDARY_RADIUS_M,
        "electrode_radius_m": REAL_ELECTRODE_RADIUS_M,
        "inner_radius_to_surface_m": REAL_INNER_RADIUS_M,
        "electrode_center_radius_m": REAL_INNER_RADIUS_M + REAL_ELECTRODE_RADIUS_M,
        "electrode_potentials_v": [1.0, 1.0, 1.0, 1.0],
        "outer_boundary_potential_v": 0.0,
        "minima_sort": "polar angle atan2(y,x) mapped to [0,2*pi)",
        "reference_row5_used": False,
    }


def _dataset_readme(
    result: SyntheticDatasetResult,
    summary: dict[str, object],
) -> str:
    statuses = summary["status_counts"]
    return f"""# Synthetic RF-trap forward dataset

This directory contains {result.config.n} deterministic forward samples drawn
with NumPy's `default_rng` seed `{result.config.seed}`. Each raw displacement
coordinate is uniform on `[-{1.0e6 * result.config.max_displacement_m:g},
+{1.0e6 * result.config.max_displacement_m:g}]` µm and is stored in metres.

## Coordinate convention

Raw inputs use Wolfram order: W1 upper-right, W2 lower-right, W3 upper-left,
and W4 lower-left. The absolute FEM displacement order is
`F1,F2,F3,F4 = -[W3,W1,W4,W2]`. All four electrodes move; the 50 mm grounded
outer circle remains fixed at the geometric origin. The electrodes are 10 mm
radius, their centre radius is 21.48 mm, all electrodes are +1 V, and the outer
boundary is 0 V.

Outputs are robust pseudopotential minima in absolute geometric-centre
coordinates, sorted by polar angle `atan2(y,x)` mapped to `[0,2*pi)`. The
practical mesh is a 500 µm central refinement with a coarse outer domain.

## Split policy

`synthetic_clean.csv` contains only samples with exactly three robust-accepted
candidates and minimum pairwise separation at least 0.15 mm.
`synthetic_rejected.csv` preserves solver failures, invalid geometry,
non-three robust topology, and `ambiguous_branch` cases. No rejected sample is
silently included in the clean file. `rejected_candidate_count` instead counts
individual candidates discarded by robust quality rules; a clean solve may have
such candidates provided exactly three candidates were robust-accepted.
`Data.txt` row 5 is not used as a training row; this dataset is sampled
independently.

Counts: clean `{summary['clean_samples']}`, rejected
`{summary['rejected_samples']}`, ambiguous branch
`{summary['ambiguous_branch_count']}`. Status counts: `{statuses}`.

## Files

- `synthetic_clean.csv`: stable 27-column training table.
- `synthetic_rejected.csv`: the same core fields plus failure/topology audit data.
- `synthetic_summary.json`: configuration, counts, convention, and runtimes.
"""


def build_parser() -> argparse.ArgumentParser:
    """Build the synthetic-dataset command-line interface with a large-run gate."""

    parser = argparse.ArgumentParser(
        prog="rf-trap-generate-dataset",
        description="Generate robust FEM samples in canonical Wolfram order.",
    )
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation_results/generated_dataset"),
    )
    parser.add_argument("--max-displacement-um", type=float, default=500.0)
    parser.add_argument("--mesh-mode", choices=("practical",), default="practical")
    parser.add_argument(
        "--allow-large-n",
        action="store_true",
        help="acknowledge that n > 1000 may take many hours",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=3,
        help="maximum concurrent fresh FEM subprocesses",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate the requested bounded dataset and print split statistics."""

    arguments = build_parser().parse_args(argv)
    config = SyntheticDatasetConfig(
        n=arguments.n,
        seed=arguments.seed,
        max_displacement_m=arguments.max_displacement_um * 1.0e-6,
        mesh_mode=arguments.mesh_mode,
        batch_size=arguments.batch_size,
        allow_large_n=arguments.allow_large_n,
    )
    update_stride = max(1, config.n // 20)

    def progress(completed: int, total: int, elapsed: float) -> None:
        if completed == total or completed % update_stride == 0:
            print(
                f"completed={completed}/{total} elapsed_seconds={elapsed:.3f}",
                flush=True,
            )

    result = generate_synthetic_dataset(config, progress_callback=progress)
    paths = write_synthetic_dataset(result, arguments.output_dir)
    statuses = Counter(item.status for item in result.records)
    print(f"requested={config.n}")
    print(f"clean={len(result.clean_records)}")
    print(f"rejected={len(result.rejected_records)}")
    print(f"ambiguous_branch={statuses.get('ambiguous_branch', 0)}")
    print(f"runtime_seconds={result.runtime_seconds:.3f}")
    print(f"clean_csv={paths.clean_csv}")
    print(f"rejected_csv={paths.rejected_csv}")
    print(f"summary_json={paths.summary_json}")
    print(f"readme={paths.readme_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
