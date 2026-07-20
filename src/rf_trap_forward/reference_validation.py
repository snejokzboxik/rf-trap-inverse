"""Benchmark the FEM forward solver against the supplied reference dataset."""

from __future__ import annotations

import argparse
import csv
import pickle
import subprocess
import sys
import textwrap
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment

from .config import ForwardModelConfig
from .dataset import ReferenceDataset, load_reference_dataset
from .demo import demonstrator_config
from .forward import ForwardModelResult


@dataclass(frozen=True)
class ForwardObservation:
    """Serializable outputs needed from one forward solve."""

    minima_positions_m: NDArray[np.float64]
    hessian_validated_candidates: int
    node_count: int
    triangle_count: int
    relative_free_residual: float
    valid_coarse_points: int
    coarse_candidates: int
    refined_candidates: int
    unique_candidates: int

    def __post_init__(self) -> None:
        """Validate observation dimensions and scalar diagnostics."""

        positions = np.asarray(self.minima_positions_m, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("minima_positions_m must have shape (n, 2)")
        if not np.all(np.isfinite(positions)):
            raise ValueError("minima_positions_m must be finite")
        counts = (
            self.hessian_validated_candidates,
            self.node_count,
            self.triangle_count,
            self.valid_coarse_points,
            self.coarse_candidates,
            self.refined_candidates,
            self.unique_candidates,
        )
        if any(value < 0 for value in counts):
            raise ValueError("forward-observation counts must be non-negative")
        if not np.isfinite(self.relative_free_residual):
            raise ValueError("relative_free_residual must be finite")
        object.__setattr__(self, "minima_positions_m", positions.copy())


@dataclass(frozen=True)
class ForwardFailure:
    """Serializable description of a failed forward solve."""

    error_type: str
    error_message: str


@dataclass(frozen=True)
class MinimumMatch:
    """One reference/computed minimum pair from spatial assignment."""

    reference_index: int
    computed_index: int
    reference_position_m: NDArray[np.float64]
    computed_position_m: NDArray[np.float64]
    delta_m: NDArray[np.float64]
    distance_m: float


@dataclass(frozen=True)
class ReferenceValidationRow:
    """Inputs, solver outcome, and matched errors for one source row."""

    row_number: int
    raw_displacements_m: NDArray[np.float64]
    relative_displacements_m: NDArray[np.float64]
    reference_minima_absolute_m: NDArray[np.float64]
    reference_minima_relative_m: NDArray[np.float64]
    status: str
    observation: ForwardObservation | None
    matches: tuple[MinimumMatch, ...]
    error_type: str = ""
    error_message: str = ""

    @property
    def completed(self) -> bool:
        """Return whether three computed minima were matched successfully."""

        return self.status == "ok" and len(self.matches) == 3

    @property
    def exactly_three_physical_minima(self) -> bool:
        """Return whether exactly three pre-selection Hessian-valid minima existed."""

        return bool(
            self.completed
            and self.observation is not None
            and self.observation.hessian_validated_candidates == 3
        )

    def error_distances_m(self) -> NDArray[np.float64]:
        """Return spatial assignment errors for this row."""

        return np.asarray([match.distance_m for match in self.matches], dtype=float)


@dataclass(frozen=True)
class ReferenceValidationSummary:
    """Aggregate metrics over all successfully matched minima."""

    selected_rows: int
    completed_rows: int
    failed_rows: int
    rows_with_exactly_three_physical_minima: int
    matched_minima: int
    mean_error_m: float
    median_error_m: float
    maximum_error_m: float
    percentile_95_error_m: float


@dataclass(frozen=True)
class ReferenceValidationReport:
    """Complete reference benchmark and the FEM configuration used for it."""

    source_path: Path | None
    model_config: ForwardModelConfig
    rows: tuple[ReferenceValidationRow, ...]

    def summary(self) -> ReferenceValidationSummary:
        """Compute aggregate error and completion metrics."""

        errors = np.asarray(
            [match.distance_m for row in self.rows for match in row.matches],
            dtype=float,
        )
        metrics = _error_metrics(errors)
        completed = sum(row.completed for row in self.rows)
        return ReferenceValidationSummary(
            selected_rows=len(self.rows),
            completed_rows=completed,
            failed_rows=len(self.rows) - completed,
            rows_with_exactly_three_physical_minima=sum(
                row.exactly_three_physical_minima for row in self.rows
            ),
            matched_minima=int(errors.size),
            mean_error_m=metrics[0],
            median_error_m=metrics[1],
            maximum_error_m=metrics[2],
            percentile_95_error_m=metrics[3],
        )


@dataclass(frozen=True)
class ReferenceValidationOutputPaths:
    """Artifacts written for one reference-validation report."""

    rows_csv: Path
    minima_csv: Path
    markdown_report: Path
    plot_paths: tuple[Path, ...]


ForwardRunner = Callable[
    [ArrayLike, ForwardModelConfig],
    ForwardModelResult | ForwardObservation,
]


def select_reference_rows(
    row_count: int,
    *,
    start_row: int | None = None,
    end_row: int | None = None,
    random_count: int | None = None,
    random_seed: int = 1,
) -> tuple[int, ...]:
    """Select one-based source rows, defaulting to the first ten.

    If ``random_count`` is supplied without a range, sampling uses the full
    dataset. If a range is supplied, sampling is restricted to that inclusive
    range. Returned row numbers are sorted for deterministic reporting.
    """

    if row_count <= 0:
        raise ValueError("row_count must be positive")
    no_explicit_range = start_row is None and end_row is None
    if no_explicit_range and random_count is None:
        start, end = 1, min(10, row_count)
    else:
        start = 1 if start_row is None else start_row
        end = row_count if end_row is None else end_row
    if start < 1 or end < start or end > row_count:
        raise ValueError("row range must be within the one-based dataset bounds")
    candidates = np.arange(start, end + 1, dtype=int)
    if random_count is None:
        return tuple(int(value) for value in candidates)
    if random_count <= 0 or random_count > candidates.size:
        raise ValueError("random_count must fit within the selected row range")
    if random_seed < 0:
        raise ValueError("random_seed must be non-negative")
    generator = np.random.default_rng(random_seed)
    selected = np.sort(generator.choice(candidates, size=random_count, replace=False))
    return tuple(int(value) for value in selected)


def match_minima_by_distance(
    reference_positions_m: ArrayLike,
    computed_positions_m: ArrayLike,
) -> tuple[MinimumMatch, ...]:
    """Match equal-size point sets using minimum-total-distance assignment."""

    reference = _point_array("reference_positions_m", reference_positions_m)
    computed = _point_array("computed_positions_m", computed_positions_m)
    if reference.shape[0] != computed.shape[0] or reference.shape[0] == 0:
        raise ValueError("reference and computed point sets must have equal nonzero size")
    cost = np.linalg.norm(
        reference[:, np.newaxis, :] - computed[np.newaxis, :, :],
        axis=2,
    )
    reference_indices, computed_indices = linear_sum_assignment(cost)
    matches = []
    for reference_index, computed_index in zip(
        reference_indices,
        computed_indices,
        strict=True,
    ):
        delta = computed[computed_index] - reference[reference_index]
        matches.append(
            MinimumMatch(
                reference_index=int(reference_index) + 1,
                computed_index=int(computed_index) + 1,
                reference_position_m=reference[reference_index].copy(),
                computed_position_m=computed[computed_index].copy(),
                delta_m=delta,
                distance_m=float(np.linalg.norm(delta)),
            )
        )
    return tuple(matches)


def forward_observation_from_result(result: ForwardModelResult) -> ForwardObservation:
    """Extract the serializable validation subset of a forward-model result."""

    diagnostics = result.minima_diagnostics
    return ForwardObservation(
        minima_positions_m=result.minima_positions_m(),
        hessian_validated_candidates=diagnostics.hessian_validated_candidates,
        node_count=result.trap_mesh.number_of_nodes,
        triangle_count=result.trap_mesh.number_of_triangles,
        relative_free_residual=result.fem_solution.relative_free_residual,
        valid_coarse_points=diagnostics.valid_coarse_points,
        coarse_candidates=diagnostics.coarse_candidates,
        refined_candidates=diagnostics.refined_candidates,
        unique_candidates=diagnostics.unique_candidates,
    )


def run_reference_validation(
    dataset: ReferenceDataset,
    model_config: ForwardModelConfig,
    row_numbers: Iterable[int],
    *,
    runner: ForwardRunner | None = None,
) -> ReferenceValidationReport:
    """Run and compare selected rows in the electrode-1 reference frame.

    Production calls isolate each Gmsh solve in a fresh interpreter. Supplying a
    runner executes in process and is intended for tests or instrumentation.
    """

    selected = tuple(int(value) for value in row_numbers)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("row_numbers must be a nonempty sequence without duplicates")
    if any(value < 1 or value > dataset.row_count for value in selected):
        raise ValueError("row_numbers contains an out-of-range row")
    rows = []
    for row_number in selected:
        row_index = row_number - 1
        raw_displacements = dataset.raw_displacements_m[row_index]
        relative_displacements = dataset.relative_displacements_flat_m[row_index]
        reference_absolute = dataset.raw_minima_absolute_m[row_index]
        reference_relative = dataset.minima_relative_to_electrode1_m[row_index]
        outcome = _execute_forward(relative_displacements, model_config, runner)
        rows.append(
            _build_row_record(
                row_number,
                raw_displacements,
                relative_displacements,
                reference_absolute,
                reference_relative,
                outcome,
            )
        )
    return ReferenceValidationReport(
        source_path=dataset.source_path,
        model_config=model_config,
        rows=tuple(rows),
    )


def write_reference_validation_outputs(
    report: ReferenceValidationReport,
    output_directory: str | Path,
) -> ReferenceValidationOutputPaths:
    """Write per-row/per-minimum CSVs, Markdown, and one plot per row."""

    output = Path(output_directory)
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    paths = ReferenceValidationOutputPaths(
        rows_csv=output / "reference_validation_rows.csv",
        minima_csv=output / "reference_validation_minima.csv",
        markdown_report=output / "reference_validation_report.md",
        plot_paths=tuple(plots / f"row_{row.row_number:04d}.png" for row in report.rows),
    )
    _write_csv(paths.rows_csv, _row_fieldnames(), _row_csv_records(report))
    _write_csv(paths.minima_csv, _minimum_fieldnames(), _minimum_csv_records(report))
    paths.markdown_report.write_text(_markdown_report(report), encoding="utf-8")
    for row, plot_path in zip(report.rows, paths.plot_paths, strict=True):
        _write_row_plot(row, report.model_config, plot_path)
    return paths


def _execute_forward(
    displacements_m: NDArray[np.float64],
    config: ForwardModelConfig,
    runner: ForwardRunner | None,
) -> ForwardObservation | ForwardFailure:
    if runner is None:
        return _run_isolated_forward(displacements_m, config)
    try:
        result = runner(displacements_m, config)
        if isinstance(result, ForwardObservation):
            return result
        return forward_observation_from_result(result)
    except Exception as error:  # validation must retain failed benchmark rows
        return ForwardFailure(type(error).__name__, str(error))


def _run_isolated_forward(
    displacements_m: NDArray[np.float64],
    config: ForwardModelConfig,
) -> ForwardObservation | ForwardFailure:
    completed = subprocess.run(
        [sys.executable, "-m", "rf_trap_forward._reference_validation_worker"],
        input=pickle.dumps((displacements_m, config)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        return ForwardFailure("WorkerProcessError", message or "isolated worker failed")
    try:
        outcome = pickle.loads(completed.stdout)
    except Exception as error:
        return ForwardFailure("WorkerProtocolError", str(error))
    if not isinstance(outcome, (ForwardObservation, ForwardFailure)):
        return ForwardFailure("WorkerProtocolError", "worker returned an invalid object")
    return outcome


def _build_row_record(
    row_number: int,
    raw_displacements_m: NDArray[np.float64],
    relative_displacements_m: NDArray[np.float64],
    reference_absolute_m: NDArray[np.float64],
    reference_relative_m: NDArray[np.float64],
    outcome: ForwardObservation | ForwardFailure,
) -> ReferenceValidationRow:
    if isinstance(outcome, ForwardFailure):
        return ReferenceValidationRow(
            row_number=row_number,
            raw_displacements_m=raw_displacements_m.copy(),
            relative_displacements_m=relative_displacements_m.copy(),
            reference_minima_absolute_m=reference_absolute_m.copy(),
            reference_minima_relative_m=reference_relative_m.copy(),
            status="forward-failed",
            observation=None,
            matches=(),
            error_type=outcome.error_type,
            error_message=outcome.error_message,
        )
    if outcome.minima_positions_m.shape != (3, 2):
        return ReferenceValidationRow(
            row_number=row_number,
            raw_displacements_m=raw_displacements_m.copy(),
            relative_displacements_m=relative_displacements_m.copy(),
            reference_minima_absolute_m=reference_absolute_m.copy(),
            reference_minima_relative_m=reference_relative_m.copy(),
            status="unexpected-minimum-count",
            observation=outcome,
            matches=(),
            error_type="UnexpectedMinimumCount",
            error_message=f"computed {outcome.minima_positions_m.shape[0]} minima; expected 3",
        )
    matches = match_minima_by_distance(reference_relative_m, outcome.minima_positions_m)
    return ReferenceValidationRow(
        row_number=row_number,
        raw_displacements_m=raw_displacements_m.copy(),
        relative_displacements_m=relative_displacements_m.copy(),
        reference_minima_absolute_m=reference_absolute_m.copy(),
        reference_minima_relative_m=reference_relative_m.copy(),
        status="ok",
        observation=outcome,
        matches=matches,
    )


def _point_array(name: str, values: ArrayLike) -> NDArray[np.float64]:
    points = np.asarray(values, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must have finite shape (n, 2)")
    return points


def _error_metrics(errors_m: NDArray[np.float64]) -> tuple[float, float, float, float]:
    if errors_m.size == 0:
        return (float("nan"),) * 4
    return (
        float(np.mean(errors_m)),
        float(np.median(errors_m)),
        float(np.max(errors_m)),
        float(np.percentile(errors_m, 95.0)),
    )


def _row_fieldnames() -> list[str]:
    fields = ["row_number", "status", "error_type", "error_message"]
    for electrode in range(1, 5):
        fields.extend([f"raw_d{electrode}_x_m", f"raw_d{electrode}_y_m"])
    for electrode in range(2, 5):
        fields.extend([f"relative_d{electrode}_x_m", f"relative_d{electrode}_y_m"])
    fields.extend(
        [
            "computed_minimum_count",
            "hessian_validated_candidates",
            "exactly_three_physical_minima",
            "node_count",
            "triangle_count",
            "relative_free_residual",
            "valid_coarse_points",
            "coarse_candidates",
            "refined_candidates",
            "unique_candidates",
            "mean_error_um",
            "mean_error_mm",
            "median_error_um",
            "median_error_mm",
            "maximum_error_um",
            "maximum_error_mm",
        ]
    )
    return fields


def _row_csv_records(report: ReferenceValidationReport) -> list[dict[str, object]]:
    records = []
    for row in report.rows:
        record: dict[str, object] = {
            "row_number": row.row_number,
            "status": row.status,
            "error_type": row.error_type,
            "error_message": row.error_message,
            "exactly_three_physical_minima": row.exactly_three_physical_minima,
        }
        _add_pairs(record, "raw_d", row.raw_displacements_m, 1)
        _add_pairs(record, "relative_d", row.relative_displacements_m.reshape(3, 2), 2)
        observation = row.observation
        record.update(
            {
                "computed_minimum_count": 0 if observation is None else observation.minima_positions_m.shape[0],
                "hessian_validated_candidates": "" if observation is None else observation.hessian_validated_candidates,
                "node_count": "" if observation is None else observation.node_count,
                "triangle_count": "" if observation is None else observation.triangle_count,
                "relative_free_residual": "" if observation is None else observation.relative_free_residual,
                "valid_coarse_points": "" if observation is None else observation.valid_coarse_points,
                "coarse_candidates": "" if observation is None else observation.coarse_candidates,
                "refined_candidates": "" if observation is None else observation.refined_candidates,
                "unique_candidates": "" if observation is None else observation.unique_candidates,
            }
        )
        metrics = _error_metrics(row.error_distances_m())
        for name, value in zip(("mean", "median", "maximum"), metrics[:3], strict=True):
            record[f"{name}_error_um"] = "" if not np.isfinite(value) else value * 1.0e6
            record[f"{name}_error_mm"] = "" if not np.isfinite(value) else value * 1.0e3
        records.append(record)
    return records


def _minimum_fieldnames() -> list[str]:
    return [
        "row_number",
        "reference_index",
        "computed_index",
        "reference_absolute_x_m",
        "reference_absolute_y_m",
        "reference_relative_x_m",
        "reference_relative_y_m",
        "computed_relative_x_m",
        "computed_relative_y_m",
        "computed_absolute_x_m",
        "computed_absolute_y_m",
        "delta_x_m",
        "delta_y_m",
        "error_m",
        "error_um",
        "error_mm",
    ]


def _minimum_csv_records(report: ReferenceValidationReport) -> list[dict[str, object]]:
    records = []
    for row in report.rows:
        electrode1 = row.raw_displacements_m[0]
        for match in row.matches:
            reference_absolute = row.reference_minima_absolute_m[match.reference_index - 1]
            computed_absolute = match.computed_position_m + electrode1
            records.append(
                {
                    "row_number": row.row_number,
                    "reference_index": match.reference_index,
                    "computed_index": match.computed_index,
                    "reference_absolute_x_m": reference_absolute[0],
                    "reference_absolute_y_m": reference_absolute[1],
                    "reference_relative_x_m": match.reference_position_m[0],
                    "reference_relative_y_m": match.reference_position_m[1],
                    "computed_relative_x_m": match.computed_position_m[0],
                    "computed_relative_y_m": match.computed_position_m[1],
                    "computed_absolute_x_m": computed_absolute[0],
                    "computed_absolute_y_m": computed_absolute[1],
                    "delta_x_m": match.delta_m[0],
                    "delta_y_m": match.delta_m[1],
                    "error_m": match.distance_m,
                    "error_um": match.distance_m * 1.0e6,
                    "error_mm": match.distance_m * 1.0e3,
                }
            )
    return records


def _add_pairs(
    record: dict[str, object],
    prefix: str,
    pairs: NDArray[np.float64],
    start_index: int,
) -> None:
    for offset, pair in enumerate(pairs):
        index = start_index + offset
        record[f"{prefix}{index}_x_m"] = pair[0]
        record[f"{prefix}{index}_y_m"] = pair[1]


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    records: Iterable[dict[str, object]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _markdown_report(report: ReferenceValidationReport) -> str:
    summary = report.summary()
    config = report.model_config
    diagnostics = _scale_diagnostics(report)
    status_counts = Counter(row.status for row in report.rows)
    lines = [
        "# FEM-to-reference validation report",
        "",
        "## Comparison convention",
        "",
        "The FEM fixes electrode 1. Each source row is therefore translated by",
        "`-d1`: solver inputs are `(d2-d1, d3-d1, d4-d1)` and reference minima",
        "are compared as `minimum_absolute-d1`. The source ordering is retained",
        "for identifiers, but pairing uses minimum-total-distance assignment.",
        "",
        "All calculations use metres. Tables display errors in both micrometres",
        "and millimetres. Failed rows are retained and excluded from error metrics.",
        "",
        "## Summary",
        "",
        f"- Selected rows: `{summary.selected_rows}`",
        f"- Completed rows: `{summary.completed_rows}`",
        f"- Failed/incomplete rows: `{summary.failed_rows}`",
        f"- Rows with exactly three pre-selection Hessian-valid minima: `{summary.rows_with_exactly_three_physical_minima}`",
        f"- Matched minima: `{summary.matched_minima}`",
        f"- Mean error: `{_format_distance(summary.mean_error_m)}`",
        f"- Median error: `{_format_distance(summary.median_error_m)}`",
        f"- Maximum error: `{_format_distance(summary.maximum_error_m)}`",
        f"- 95th-percentile error: `{_format_distance(summary.percentile_95_error_m)}`",
        f"- Row statuses: `{dict(sorted(status_counts.items()))}`",
        "",
        "## Scale and boundary diagnostics",
        "",
        f"- FEM nominal centre radius: `{np.linalg.norm(config.geometry.nominal_centers_m[0]) * 1.0e3:.6g} mm`",
        f"- FEM electrode radius: `{config.geometry.electrode_radius_m * 1.0e3:.6g} mm`",
        f"- FEM outer-boundary radius: `{config.geometry.outer_radius_m * 1.0e3:.6g} mm`",
        f"- FEM minima-search half-extent: `{config.minima.search_half_extent_m * 1.0e3:.6g} mm`",
        f"- Reference minima outside the search square: `{diagnostics['outside_search']}` of `{diagnostics['reference_count']}`",
        f"- Reference minima outside the FEM outer circle: `{diagnostics['outside_outer']}` of `{diagnostics['reference_count']}`",
        f"- Reference radial-distance median/range: `{diagnostics['reference_median_radius_m'] * 1.0e3:.6g} mm` / `{diagnostics['reference_min_radius_m'] * 1.0e3:.6g}` to `{diagnostics['reference_max_radius_m'] * 1.0e3:.6g} mm`",
        f"- Computed radial-distance range: `{_format_range_mm(diagnostics['computed_min_radius_m'], diagnostics['computed_max_radius_m'])}`",
        f"- Maximum electrode-1 translation applied to the selected rows: `{diagnostics['maximum_d1_radius_m'] * 1.0e3:.6g} mm`",
        "",
        "## Per-row errors",
        "",
        "| row | status | FEM minima | Hessian-valid | exactly 3 physical | mean error (µm / mm) | median error (µm / mm) | max error (µm / mm) | failure |",
        "|---:|:---|---:|---:|:---:|---:|---:|---:|:---|",
    ]
    for row in report.rows:
        observation = row.observation
        errors = _error_metrics(row.error_distances_m())
        lines.append(
            f"| {row.row_number} | {row.status} "
            f"| {0 if observation is None else observation.minima_positions_m.shape[0]} "
            f"| {'-' if observation is None else observation.hessian_validated_candidates} "
            f"| {_yes_no(row.exactly_three_physical_minima)} "
            f"| {_format_distance(errors[0])} | {_format_distance(errors[1])} "
            f"| {_format_distance(errors[2])} "
            f"| {_escape_markdown(_short_failure(row)).strip(': ')} |"
        )
    lines.extend(
        [
            "",
            "## Per-minimum spatial assignment",
            "",
            "| row | reference | computed | reference relative (mm) | computed relative (mm) | error (µm / mm) |",
            "|---:|---:|---:|:---|:---|---:|",
        ]
    )
    for row in report.rows:
        for match in row.matches:
            lines.append(
                f"| {row.row_number} | {match.reference_index} | {match.computed_index} "
                f"| ({match.reference_position_m[0] * 1.0e3:.6g}, {match.reference_position_m[1] * 1.0e3:.6g}) "
                f"| ({match.computed_position_m[0] * 1.0e3:.6g}, {match.computed_position_m[1] * 1.0e3:.6g}) "
                f"| {_format_distance(match.distance_m)} |"
            )
    lines.extend(
        [
            "",
            "## Diagnostic interpretation",
            "",
            "- **Electrode numbering:** source numbering is assumed to match FEM",
            "  numbering; the supplied data do not independently establish this map.",
            "- **Coordinate origin / absolute versus electrode-1-relative:** the",
            "  benchmark explicitly applies the required electrode-1 translation.",
            "  Its magnitude is reported above, so remaining multi-millimetre scale",
            "  disagreement cannot be attributed solely to this convention.",
            "- **Polarity convention:** the current FEM drives four electrodes at",
            "  equal phase, whereas the reference article describes an eight-rod",
            "  alternating-polarity octupole. Equivalence is not established.",
            "- **Geometry scale:** the FEM radius and nominal centres are explicitly",
            "  provisional. Reference minima several millimetres from the origin are",
            "  incompatible with treating the current demonstrator dimensions as",
            "  validated physical geometry.",
            "- **Outer boundary and search region:** reference points outside the",
            "  configured search square cannot be recovered by this solver run; points",
            "  outside the finite outer circle are outside the modeled vacuum domain.",
            "",
            "These results validate the current implementation against the supplied",
            "data only at the stated conventions and provisional geometry. They do not",
            "authorize synthetic dataset generation unless the measured errors and",
            "failure modes are resolved with the actual electrode geometry, polarity,",
            "numbering, and boundary/search scales.",
            "",
        ]
    )
    return "\n".join(lines)


def _scale_diagnostics(report: ReferenceValidationReport) -> dict[str, float | int]:
    reference = np.vstack([row.reference_minima_relative_m for row in report.rows])
    reference_radii = np.linalg.norm(reference, axis=1)
    computed_arrays = [
        row.observation.minima_positions_m
        for row in report.rows
        if row.observation is not None and row.observation.minima_positions_m.size
    ]
    computed_radii = (
        np.linalg.norm(np.vstack(computed_arrays), axis=1)
        if computed_arrays
        else np.asarray([], dtype=float)
    )
    extent = report.model_config.minima.search_half_extent_m
    outer = report.model_config.geometry.outer_radius_m
    return {
        "reference_count": int(reference.shape[0]),
        "outside_search": int(np.count_nonzero(np.any(np.abs(reference) > extent, axis=1))),
        "outside_outer": int(np.count_nonzero(reference_radii >= outer)),
        "reference_min_radius_m": float(np.min(reference_radii)),
        "reference_median_radius_m": float(np.median(reference_radii)),
        "reference_max_radius_m": float(np.max(reference_radii)),
        "computed_min_radius_m": float(np.min(computed_radii)) if computed_radii.size else float("nan"),
        "computed_max_radius_m": float(np.max(computed_radii)) if computed_radii.size else float("nan"),
        "maximum_d1_radius_m": max(
            float(np.linalg.norm(row.raw_displacements_m[0])) for row in report.rows
        ),
    }


def _format_distance(value_m: float) -> str:
    if not np.isfinite(value_m):
        return "n/a"
    return f"{value_m * 1.0e6:.6g} µm / {value_m * 1.0e3:.6g} mm"


def _format_range_mm(minimum_m: float, maximum_m: float) -> str:
    if not np.isfinite(minimum_m) or not np.isfinite(maximum_m):
        return "n/a"
    return f"{minimum_m * 1.0e3:.6g} to {maximum_m * 1.0e3:.6g} mm"


def _escape_markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _short_failure(row: ReferenceValidationRow) -> str:
    if not row.error_type and not row.error_message:
        return ""
    message = row.error_message.split("; diagnostics=", maxsplit=1)[0]
    value = f"{row.error_type}: {message}".strip(": ")
    return value if len(value) <= 180 else value[:177] + "..."


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _write_row_plot(
    row: ReferenceValidationRow,
    config: ForwardModelConfig,
    path: Path,
) -> None:
    figure = Figure(figsize=(6.6, 6.0), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    reference_mm = row.reference_minima_relative_m * 1.0e3
    axis.scatter(
        reference_mm[:, 0],
        reference_mm[:, 1],
        marker="x",
        s=85,
        linewidths=2.0,
        color="C0",
        label="reference",
        zorder=4,
    )
    for index, point in enumerate(reference_mm, start=1):
        axis.annotate(f"R{index}", point, xytext=(5, 5), textcoords="offset points")
    if row.observation is not None:
        computed_mm = row.observation.minima_positions_m * 1.0e3
        axis.scatter(
            computed_mm[:, 0],
            computed_mm[:, 1],
            marker="o",
            s=55,
            facecolors="none",
            edgecolors="C1",
            linewidths=1.8,
            label="FEM",
            zorder=5,
        )
        for index, point in enumerate(computed_mm, start=1):
            axis.annotate(f"F{index}", point, xytext=(5, -13), textcoords="offset points")
        for match in row.matches:
            endpoints = np.vstack(
                (match.reference_position_m, match.computed_position_m)
            ) * 1.0e3
            axis.plot(endpoints[:, 0], endpoints[:, 1], color="0.45", linewidth=0.8)
    search_extent_mm = config.minima.search_half_extent_m * 1.0e3
    axis.add_patch(
        Rectangle(
            (-search_extent_mm, -search_extent_mm),
            2.0 * search_extent_mm,
            2.0 * search_extent_mm,
            fill=False,
            linestyle=":",
            color="C2",
            label="search boundary",
        )
    )
    outer_radius_mm = config.geometry.outer_radius_m * 1.0e3
    axis.add_patch(
        Circle(
            (0.0, 0.0),
            outer_radius_mm,
            fill=False,
            linestyle="--",
            color="0.35",
            label="outer boundary",
        )
    )
    axis.set_title(f"Reference row {row.row_number}: {row.status}")
    axis.set_xlabel("x relative to electrode 1 (mm)")
    axis.set_ylabel("y relative to electrode 1 (mm)")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    axis.margins(0.12)
    if row.error_message:
        axis.text(
            0.02,
            0.02,
            textwrap.fill(_short_failure(row), width=68),
            transform=axis.transAxes,
            fontsize=8,
            va="bottom",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )
    figure.savefig(path, dpi=180, bbox_inches="tight")


def build_parser() -> argparse.ArgumentParser:
    """Build the reference-validation command-line parser."""

    parser = argparse.ArgumentParser(
        prog="rf-trap-reference-validation",
        description="Compare selected reference rows with isolated FEM forward solves.",
    )
    parser.add_argument("input", nargs="?", type=Path, default=Path("Data.txt"))
    parser.add_argument("--start-row", type=int)
    parser.add_argument("--end-row", type=int)
    parser.add_argument("--random-count", type=int)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("validation_results") / "milestone_4",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the configured reference benchmark and write its artifacts."""

    arguments = build_parser().parse_args(argv)
    dataset = load_reference_dataset(arguments.input)
    rows = select_reference_rows(
        dataset.row_count,
        start_row=arguments.start_row,
        end_row=arguments.end_row,
        random_count=arguments.random_count,
        random_seed=arguments.random_seed,
    )
    report = run_reference_validation(dataset, demonstrator_config(), rows)
    paths = write_reference_validation_outputs(report, arguments.output_directory)
    summary = report.summary()
    print(f"rows: {rows}")
    print(f"completed rows: {summary.completed_rows}/{summary.selected_rows}")
    print(
        "rows with exactly three physical minima: "
        f"{summary.rows_with_exactly_three_physical_minima}/{summary.selected_rows}"
    )
    print(f"mean error: {_format_distance(summary.mean_error_m)}")
    print(f"median error: {_format_distance(summary.median_error_m)}")
    print(f"maximum error: {_format_distance(summary.maximum_error_m)}")
    print(f"95th-percentile error: {_format_distance(summary.percentile_95_error_m)}")
    print(f"row CSV: {paths.rows_csv}")
    print(f"minimum CSV: {paths.minima_csv}")
    print(f"report: {paths.markdown_report}")
    print(f"plots: {len(paths.plot_paths)} under {paths.plot_paths[0].parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
