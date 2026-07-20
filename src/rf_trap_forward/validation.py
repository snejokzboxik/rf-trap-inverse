"""Convergence studies and reproducible reports for the forward solver."""

from __future__ import annotations

import csv
import pickle
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment

from .config import ForwardModelConfig, MeshConfig, displacement_vector_m
from .forward import ForwardModelResult, run_forward_model
from .minima import LocalMinimum, MinimaDiagnostics

ComparisonAxis = Literal["mesh_size_m", "outer_radius_m"]
ForwardRunner = Callable[[ArrayLike, ForwardModelConfig], ForwardModelResult]


@dataclass(frozen=True)
class ConvergenceStudyConfig:
    """Independent numerical parameters for a full factorial convergence sweep."""

    mesh_sizes_m: tuple[float, ...]
    outer_radii_m: tuple[float, ...]
    coordinate_tolerance_m: float = 1.0e-5
    expected_minima: int = 3

    def __post_init__(self) -> None:
        """Validate refinement values and comparison controls."""

        _validate_parameter_values("mesh_sizes_m", self.mesh_sizes_m)
        _validate_parameter_values("outer_radii_m", self.outer_radii_m)
        if not np.isfinite(self.coordinate_tolerance_m) or self.coordinate_tolerance_m <= 0.0:
            raise ValueError("coordinate_tolerance_m must be finite and positive")
        if self.expected_minima <= 0:
            raise ValueError("expected_minima must be positive")


@dataclass(frozen=True)
class ConvergenceRunRecord:
    """Serializable numerical diagnostics for one mesh/radius combination."""

    mesh_size_m: float
    boundary_tolerance_m: float
    gmsh_algorithm: int
    random_seed: int
    random_factor: float
    gmsh_reproducible: bool
    outer_radius_m: float
    node_count: int
    triangle_count: int
    relative_free_residual: float
    electrode_boundary_error_v: float
    outer_boundary_error_v: float
    minima_diagnostics: MinimaDiagnostics
    minima: tuple[LocalMinimum, ...]

    def minima_positions_m(self) -> NDArray[np.float64]:
        """Return stored angle-sorted minimum positions as an ``(n, 2)`` array."""

        if not self.minima:
            return np.empty((0, 2), dtype=float)
        return np.vstack([minimum.position_m for minimum in self.minima])


@dataclass(frozen=True)
class CoordinateComparison:
    """One spatially matched minimum displacement between successive runs."""

    axis: ComparisonAxis
    fixed_parameter_m: float
    previous_parameter_m: float
    current_parameter_m: float
    minimum_index: int
    previous_position_m: NDArray[np.float64]
    current_position_m: NDArray[np.float64]
    delta_position_m: NDArray[np.float64]
    distance_m: float


@dataclass(frozen=True)
class ConvergenceReport:
    """Complete run data, comparisons, and validation decisions for one sweep."""

    displacements_m: NDArray[np.float64]
    study_config: ConvergenceStudyConfig
    runs: tuple[ConvergenceRunRecord, ...]
    comparisons: tuple[CoordinateComparison, ...]
    three_minimum_structure_stable: bool
    coordinate_changes_within_tolerance: bool

    def maximum_coordinate_change_m(self, axis: ComparisonAxis | None = None) -> float:
        """Return the largest matched coordinate displacement, optionally by axis."""

        selected = [
            item.distance_m
            for item in self.comparisons
            if axis is None or item.axis == axis
        ]
        return max(selected, default=0.0)


@dataclass(frozen=True)
class ConvergenceOutputPaths:
    """Paths written by :func:`write_convergence_outputs`."""

    runs_csv: Path
    comparisons_csv: Path
    markdown_report: Path
    mesh_refinement_plot: Path
    outer_radius_plot: Path


def run_convergence_study(
    displacements_m: ArrayLike,
    base_config: ForwardModelConfig,
    study_config: ConvergenceStudyConfig,
    runner: ForwardRunner | None = None,
) -> ConvergenceReport:
    """Run every mesh-size/outer-radius combination and build a report.

    The production default executes :func:`run_forward_model` in a fresh direct
    interpreter subprocess for every case so Gmsh history cannot couple runs.
    Supplying ``runner`` uses in-process execution for fast synthetic tests or
    specialized instrumentation.
    """

    displacement = displacement_vector_m(displacements_m)
    cases = _convergence_cases(study_config)
    if runner is None:
        records = _run_isolated_cases(displacement, base_config, cases)
    else:
        records = _run_in_process_cases(displacement, base_config, cases, runner)
    return build_convergence_report(displacement, study_config, records)


def make_convergence_run_record(
    result: ForwardModelResult,
    mesh_config: MeshConfig,
) -> ConvergenceRunRecord:
    """Extract serializable diagnostics from one forward-model result."""

    solution = result.fem_solution
    return ConvergenceRunRecord(
        mesh_size_m=mesh_config.characteristic_length_m,
        boundary_tolerance_m=mesh_config.boundary_tolerance_m,
        gmsh_algorithm=mesh_config.gmsh_algorithm,
        random_seed=mesh_config.random_seed,
        random_factor=mesh_config.random_factor,
        gmsh_reproducible=mesh_config.reproducible,
        outer_radius_m=result.geometry.config.outer_radius_m,
        node_count=result.trap_mesh.number_of_nodes,
        triangle_count=result.trap_mesh.number_of_triangles,
        relative_free_residual=solution.relative_free_residual,
        electrode_boundary_error_v=solution.electrode_boundary_error_v,
        outer_boundary_error_v=solution.outer_boundary_error_v,
        minima_diagnostics=result.minima_diagnostics,
        minima=result.minima,
    )


def build_convergence_report(
    displacements_m: ArrayLike,
    study_config: ConvergenceStudyConfig,
    runs: Iterable[ConvergenceRunRecord],
) -> ConvergenceReport:
    """Build convergence comparisons and decisions from precomputed run records."""

    displacement = displacement_vector_m(displacements_m)
    ordered_runs = tuple(
        sorted(runs, key=lambda item: (item.outer_radius_m, -item.mesh_size_m))
    )
    comparisons = compare_successive_minima(ordered_runs)
    structure_stable = _has_stable_minimum_structure(
        ordered_runs,
        study_config.expected_minima,
    )
    coordinate_stable = bool(comparisons) and all(
        item.distance_m <= study_config.coordinate_tolerance_m
        for item in comparisons
    )
    return ConvergenceReport(
        displacements_m=displacement,
        study_config=study_config,
        runs=ordered_runs,
        comparisons=comparisons,
        three_minimum_structure_stable=structure_stable,
        coordinate_changes_within_tolerance=coordinate_stable,
    )


def compare_successive_minima(
    runs: Iterable[ConvergenceRunRecord],
) -> tuple[CoordinateComparison, ...]:
    """Match and compare minima along both independent refinement axes.

    Each pair uses a minimum-total-distance assignment.  Mesh sequences proceed
    from coarse to fine; outer-radius sequences proceed from smaller to larger.
    """

    run_tuple = tuple(runs)
    comparisons: list[CoordinateComparison] = []
    for fixed_value, sequence in _grouped_sequences(run_tuple, "mesh_size_m"):
        comparisons.extend(
            _compare_sequence(sequence, "mesh_size_m", fixed_value)
        )
    for fixed_value, sequence in _grouped_sequences(run_tuple, "outer_radius_m"):
        comparisons.extend(
            _compare_sequence(sequence, "outer_radius_m", fixed_value)
        )
    return tuple(comparisons)


def write_convergence_outputs(
    report: ConvergenceReport,
    output_directory: str | Path,
) -> ConvergenceOutputPaths:
    """Write CSV tables, a Markdown report, and two headless PNG plots."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = ConvergenceOutputPaths(
        runs_csv=output / "convergence_runs.csv",
        comparisons_csv=output / "coordinate_comparisons.csv",
        markdown_report=output / "convergence_report.md",
        mesh_refinement_plot=output / "mesh_refinement_minima.png",
        outer_radius_plot=output / "outer_radius_minima.png",
    )
    _write_csv(paths.runs_csv, _run_fieldnames(report), _run_rows(report))
    _write_csv(
        paths.comparisons_csv,
        _comparison_fieldnames(),
        _comparison_rows(report.comparisons),
    )
    paths.markdown_report.write_text(_markdown_report(report), encoding="utf-8")
    _write_axis_plot(report, "mesh_size_m", paths.mesh_refinement_plot)
    _write_axis_plot(report, "outer_radius_m", paths.outer_radius_plot)
    return paths


def _validate_parameter_values(name: str, values: Sequence[float]) -> None:
    if len(values) < 2:
        raise ValueError(f"{name} must contain at least two values")
    numeric = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(numeric)) or np.any(numeric <= 0.0):
        raise ValueError(f"{name} must contain finite positive values")
    if np.unique(numeric).size != numeric.size:
        raise ValueError(f"{name} must not contain duplicates")


def _replace_numerical_parameters(
    base_config: ForwardModelConfig,
    mesh_size_m: float,
    outer_radius_m: float,
) -> ForwardModelConfig:
    return replace(
        base_config,
        geometry=replace(base_config.geometry, outer_radius_m=outer_radius_m),
        mesh=replace(base_config.mesh, characteristic_length_m=mesh_size_m),
    )


def _convergence_cases(
    study_config: ConvergenceStudyConfig,
) -> list[tuple[float, float]]:
    return [
        (mesh_size_m, outer_radius_m)
        for outer_radius_m in sorted(study_config.outer_radii_m)
        for mesh_size_m in sorted(study_config.mesh_sizes_m, reverse=True)
    ]


def _run_in_process_cases(
    displacement_m: NDArray[np.float64],
    base_config: ForwardModelConfig,
    cases: Sequence[tuple[float, float]],
    runner: ForwardRunner,
) -> list[ConvergenceRunRecord]:
    records = []
    for mesh_size_m, outer_radius_m in cases:
        run_config = _replace_numerical_parameters(
            base_config,
            mesh_size_m,
            outer_radius_m,
        )
        result = runner(displacement_m, run_config)
        records.append(make_convergence_run_record(result, run_config.mesh))
    return records


def _run_isolated_cases(
    displacement_m: NDArray[np.float64],
    base_config: ForwardModelConfig,
    cases: Sequence[tuple[float, float]],
) -> list[ConvergenceRunRecord]:
    records: list[ConvergenceRunRecord] = []
    for mesh_size_m, outer_radius_m in cases:
        payload = pickle.dumps(
            (displacement_m, base_config, mesh_size_m, outer_radius_m)
        )
        completed = subprocess.run(
            [sys.executable, "-m", "rf_trap_forward._validation_worker"],
            input=payload,
            stdout=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "isolated convergence worker failed for "
                f"mesh_size_m={mesh_size_m}, outer_radius_m={outer_radius_m}"
            )
        record = pickle.loads(completed.stdout)
        if not isinstance(record, ConvergenceRunRecord):
            raise RuntimeError("isolated convergence worker returned an invalid record")
        records.append(record)
    return records


def _run_isolated_case(
    displacement_m: NDArray[np.float64],
    base_config: ForwardModelConfig,
    mesh_size_m: float,
    outer_radius_m: float,
) -> ConvergenceRunRecord:
    run_config = _replace_numerical_parameters(
        base_config,
        mesh_size_m,
        outer_radius_m,
    )
    result = run_forward_model(displacement_m, run_config)
    return make_convergence_run_record(result, run_config.mesh)


def _has_stable_minimum_structure(
    runs: Sequence[ConvergenceRunRecord],
    expected_minima: int,
) -> bool:
    if not runs:
        return False
    for run in runs:
        if len(run.minima) != expected_minima:
            return False
        if run.minima_diagnostics.hessian_validated_candidates != expected_minima:
            return False
        if any(
            minimum.hessian_eigenvalues_v2_per_m4.shape != (2,)
            or np.any(minimum.hessian_eigenvalues_v2_per_m4 <= 0.0)
            for minimum in run.minima
        ):
            return False
    return True


def _grouped_sequences(
    runs: Sequence[ConvergenceRunRecord],
    axis: ComparisonAxis,
) -> list[tuple[float, tuple[ConvergenceRunRecord, ...]]]:
    if axis == "mesh_size_m":
        fixed_values = sorted({run.outer_radius_m for run in runs})
        return [
            (
                value,
                tuple(
                    sorted(
                        (run for run in runs if run.outer_radius_m == value),
                        key=lambda item: item.mesh_size_m,
                        reverse=True,
                    )
                ),
            )
            for value in fixed_values
        ]
    fixed_values = sorted({run.mesh_size_m for run in runs}, reverse=True)
    return [
        (
            value,
            tuple(
                sorted(
                    (run for run in runs if run.mesh_size_m == value),
                    key=lambda item: item.outer_radius_m,
                )
            ),
        )
        for value in fixed_values
    ]


def _axis_parameter(run: ConvergenceRunRecord, axis: ComparisonAxis) -> float:
    return run.mesh_size_m if axis == "mesh_size_m" else run.outer_radius_m


def _compare_sequence(
    sequence: Sequence[ConvergenceRunRecord],
    axis: ComparisonAxis,
    fixed_value: float,
) -> list[CoordinateComparison]:
    comparisons: list[CoordinateComparison] = []
    for previous, current in zip(sequence, sequence[1:], strict=False):
        previous_points = previous.minima_positions_m()
        current_points = current.minima_positions_m()
        if previous_points.size == 0 or current_points.size == 0:
            continue
        cost = np.linalg.norm(
            previous_points[:, np.newaxis, :] - current_points[np.newaxis, :, :],
            axis=2,
        )
        previous_indices, current_indices = linear_sum_assignment(cost)
        for previous_index, current_index in zip(
            previous_indices,
            current_indices,
            strict=True,
        ):
            delta = current_points[current_index] - previous_points[previous_index]
            comparisons.append(
                CoordinateComparison(
                    axis=axis,
                    fixed_parameter_m=fixed_value,
                    previous_parameter_m=_axis_parameter(previous, axis),
                    current_parameter_m=_axis_parameter(current, axis),
                    minimum_index=int(previous_index) + 1,
                    previous_position_m=previous_points[previous_index],
                    current_position_m=current_points[current_index],
                    delta_position_m=delta,
                    distance_m=float(np.linalg.norm(delta)),
                )
            )
    return comparisons


def _run_fieldnames(report: ConvergenceReport) -> list[str]:
    fields = [
        "mesh_size_m",
        "boundary_tolerance_m",
        "gmsh_algorithm",
        "random_seed",
        "random_factor",
        "gmsh_reproducible",
        "outer_radius_m",
        "node_count",
        "triangle_count",
        "relative_free_residual",
        "electrode_boundary_error_v",
        "outer_boundary_error_v",
        "valid_coarse_points",
        "coarse_candidates",
        "refined_candidates",
        "unique_candidates",
        "hessian_validated_candidates",
        "minimum_count",
    ]
    for index in range(1, report.study_config.expected_minima + 1):
        fields.extend(
            [
                f"minimum_{index}_x_m",
                f"minimum_{index}_y_m",
                f"minimum_{index}_polar_angle_rad",
                f"minimum_{index}_psi_v2_per_m2",
                f"minimum_{index}_hessian_eigenvalue_1_v2_per_m4",
                f"minimum_{index}_hessian_eigenvalue_2_v2_per_m4",
                f"minimum_{index}_optimizer_succeeded",
            ]
        )
    return fields


def _run_rows(report: ConvergenceReport) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in report.runs:
        diagnostics = run.minima_diagnostics
        row: dict[str, object] = {
            "mesh_size_m": run.mesh_size_m,
            "boundary_tolerance_m": run.boundary_tolerance_m,
            "gmsh_algorithm": run.gmsh_algorithm,
            "random_seed": run.random_seed,
            "random_factor": run.random_factor,
            "gmsh_reproducible": run.gmsh_reproducible,
            "outer_radius_m": run.outer_radius_m,
            "node_count": run.node_count,
            "triangle_count": run.triangle_count,
            "relative_free_residual": run.relative_free_residual,
            "electrode_boundary_error_v": run.electrode_boundary_error_v,
            "outer_boundary_error_v": run.outer_boundary_error_v,
            "valid_coarse_points": diagnostics.valid_coarse_points,
            "coarse_candidates": diagnostics.coarse_candidates,
            "refined_candidates": diagnostics.refined_candidates,
            "unique_candidates": diagnostics.unique_candidates,
            "hessian_validated_candidates": diagnostics.hessian_validated_candidates,
            "minimum_count": len(run.minima),
        }
        for index, minimum in enumerate(run.minima, start=1):
            if index > report.study_config.expected_minima:
                break
            row.update(
                {
                    f"minimum_{index}_x_m": minimum.position_m[0],
                    f"minimum_{index}_y_m": minimum.position_m[1],
                    f"minimum_{index}_polar_angle_rad": minimum.polar_angle_rad,
                    f"minimum_{index}_psi_v2_per_m2": minimum.pseudopotential_v2_per_m2,
                    f"minimum_{index}_hessian_eigenvalue_1_v2_per_m4": minimum.hessian_eigenvalues_v2_per_m4[0],
                    f"minimum_{index}_hessian_eigenvalue_2_v2_per_m4": minimum.hessian_eigenvalues_v2_per_m4[1],
                    f"minimum_{index}_optimizer_succeeded": minimum.optimizer_succeeded,
                }
            )
        rows.append(row)
    return rows


def _comparison_fieldnames() -> list[str]:
    return [
        "axis",
        "fixed_parameter_m",
        "previous_parameter_m",
        "current_parameter_m",
        "minimum_index",
        "previous_x_m",
        "previous_y_m",
        "current_x_m",
        "current_y_m",
        "delta_x_m",
        "delta_y_m",
        "distance_m",
    ]


def _comparison_rows(
    comparisons: Iterable[CoordinateComparison],
) -> list[dict[str, object]]:
    return [
        {
            "axis": item.axis,
            "fixed_parameter_m": item.fixed_parameter_m,
            "previous_parameter_m": item.previous_parameter_m,
            "current_parameter_m": item.current_parameter_m,
            "minimum_index": item.minimum_index,
            "previous_x_m": item.previous_position_m[0],
            "previous_y_m": item.previous_position_m[1],
            "current_x_m": item.current_position_m[0],
            "current_y_m": item.current_position_m[1],
            "delta_x_m": item.delta_position_m[0],
            "delta_y_m": item.delta_position_m[1],
            "distance_m": item.distance_m,
        }
        for item in comparisons
    ]


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[dict[str, object]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _markdown_report(report: ConvergenceReport) -> str:
    mesh_max_um = report.maximum_coordinate_change_m("mesh_size_m") * 1.0e6
    outer_max_um = report.maximum_coordinate_change_m("outer_radius_m") * 1.0e6
    lines = [
        "# Forward-model convergence report",
        "",
        "The study is a full Cartesian product of the configured mesh sizes and outer radii.",
        "Stored coordinates are sorted by polar angle; successive comparisons use",
        "minimum-total-distance spatial assignment.",
        "",
        "## Summary",
        "",
        f"- Displacements (m): `{report.displacements_m.tolist()}`",
        f"- Three-minimum structure stable: **{_yes_no(report.three_minimum_structure_stable)}**",
        f"- All coordinate changes within {report.study_config.coordinate_tolerance_m * 1.0e6:.6g} µm: **{_yes_no(report.coordinate_changes_within_tolerance)}**",
        f"- Maximum mesh-refinement coordinate change: `{mesh_max_um:.6g} µm`",
        f"- Maximum outer-radius coordinate change: `{outer_max_um:.6g} µm`",
        "",
        "## Run diagnostics",
        "",
        "| h (µm) | outer radius (mm) | algorithm | seed | random factor | reproducible | nodes | triangles | relative residual | valid coarse | coarse | refined | unique | Hessian-valid | minima |",
        "|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in report.runs:
        diagnostics = run.minima_diagnostics
        lines.append(
            f"| {run.mesh_size_m * 1.0e6:.6g} | {run.outer_radius_m * 1.0e3:.6g} "
            f"| {run.gmsh_algorithm} | {run.random_seed} | {run.random_factor:.3g} "
            f"| {_yes_no(run.gmsh_reproducible)} | {run.node_count} "
            f"| {run.triangle_count} | {run.relative_free_residual:.6e} "
            f"| {diagnostics.valid_coarse_points} | {diagnostics.coarse_candidates} "
            f"| {diagnostics.refined_candidates} | {diagnostics.unique_candidates} "
            f"| {diagnostics.hessian_validated_candidates} | {len(run.minima)} |"
        )
    lines.extend(_minimum_markdown_table(report.runs))
    lines.extend(_comparison_markdown_table(report.comparisons))
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "This report validates the provisional demonstrator only. It does not establish",
            "production accuracy for the physical trap geometry, which has not yet been supplied.",
            "`Psi` is `|E|²`, not a dimensional pseudopotential energy.",
            "",
        ]
    )
    return "\n".join(lines)


def _minimum_markdown_table(runs: Sequence[ConvergenceRunRecord]) -> list[str]:
    lines = [
        "",
        "## Angle-sorted minima",
        "",
        "| h (µm) | outer radius (mm) | minimum | x (µm) | y (µm) | Psi (V²/m²) | Hessian λ1 (V²/m⁴) | Hessian λ2 (V²/m⁴) | optimizer |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for run in runs:
        for index, minimum in enumerate(run.minima, start=1):
            lines.append(
                f"| {run.mesh_size_m * 1.0e6:.6g} | {run.outer_radius_m * 1.0e3:.6g} "
                f"| {index} | {minimum.position_m[0] * 1.0e6:.8g} "
                f"| {minimum.position_m[1] * 1.0e6:.8g} "
                f"| {minimum.pseudopotential_v2_per_m2:.6e} "
                f"| {minimum.hessian_eigenvalues_v2_per_m4[0]:.6e} "
                f"| {minimum.hessian_eigenvalues_v2_per_m4[1]:.6e} "
                f"| {_yes_no(minimum.optimizer_succeeded)} |"
            )
    return lines


def _comparison_markdown_table(
    comparisons: Sequence[CoordinateComparison],
) -> list[str]:
    lines = [
        "",
        "## Successive coordinate comparisons",
        "",
        "| axis | fixed parameter (m) | previous (m) | current (m) | minimum | Δx (µm) | Δy (µm) | distance (µm) |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in comparisons:
        lines.append(
            f"| {item.axis} | {item.fixed_parameter_m:.8g} "
            f"| {item.previous_parameter_m:.8g} | {item.current_parameter_m:.8g} "
            f"| {item.minimum_index} | {item.delta_position_m[0] * 1.0e6:.8g} "
            f"| {item.delta_position_m[1] * 1.0e6:.8g} "
            f"| {item.distance_m * 1.0e6:.8g} |"
        )
    return lines


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _write_axis_plot(
    report: ConvergenceReport,
    axis: ComparisonAxis,
    path: Path,
) -> None:
    sequences = _grouped_sequences(report.runs, axis)
    minimum_count = min(
        (len(run.minima) for run in report.runs),
        default=0,
    )
    if not sequences or minimum_count == 0:
        raise ValueError("at least one minimum is required to create convergence plots")
    figure = Figure(
        figsize=(4.4 * minimum_count, 3.8 * len(sequences)),
        layout="constrained",
    )
    FigureCanvasAgg(figure)
    axes = np.asarray(
        figure.subplots(len(sequences), minimum_count, squeeze=False)
    )
    for row_index, (fixed_value, sequence) in enumerate(sequences):
        tracked = _tracked_positions(sequence)
        parameters = [_axis_parameter(run, axis) for run in sequence]
        for minimum_index in range(tracked.shape[1]):
            plot_axis = axes[row_index, minimum_index]
            points_um = tracked[:, minimum_index, :] * 1.0e6
            plot_axis.plot(
                points_um[:, 0],
                points_um[:, 1],
                marker="o",
                color=f"C{minimum_index % 10}",
                label=f"minimum {minimum_index + 1}",
            )
            for parameter_index, (point, parameter) in enumerate(
                zip(points_um, parameters, strict=True)
            ):
                plot_axis.annotate(
                    _parameter_label(parameter, axis),
                    point,
                    xytext=(5, 6 if parameter_index % 2 == 0 else -12),
                    textcoords="offset points",
                    fontsize=7,
                )
            plot_axis.set_title(
                f"{_fixed_parameter_title(fixed_value, axis)}\n"
                f"minimum {minimum_index + 1}"
            )
            plot_axis.set_xlabel("x (µm)")
            plot_axis.set_ylabel("y (µm)")
            plot_axis.set_aspect("equal", adjustable="datalim")
            plot_axis.margins(0.25)
            plot_axis.grid(True, alpha=0.3)
    figure.suptitle(_plot_title(axis))
    figure.savefig(path, dpi=180, bbox_inches="tight")


def _tracked_positions(
    sequence: Sequence[ConvergenceRunRecord],
) -> NDArray[np.float64]:
    if not sequence:
        return np.empty((0, 0, 2), dtype=float)
    minimum_count = min(len(run.minima) for run in sequence)
    if minimum_count == 0:
        return np.empty((len(sequence), 0, 2), dtype=float)
    previous = sequence[0].minima_positions_m()[:minimum_count]
    tracked = [previous]
    for run in sequence[1:]:
        current = run.minima_positions_m()
        cost = np.linalg.norm(
            previous[:, np.newaxis, :] - current[np.newaxis, :, :],
            axis=2,
        )
        previous_indices, current_indices = linear_sum_assignment(cost)
        order = np.argsort(previous_indices)
        previous = current[current_indices[order]]
        tracked.append(previous)
    return np.stack(tracked)


def _parameter_label(value_m: float, axis: ComparisonAxis) -> str:
    if axis == "mesh_size_m":
        return f"h={value_m * 1.0e6:.3g} µm"
    return f"R={value_m * 1.0e3:.3g} mm"


def _fixed_parameter_title(value_m: float, axis: ComparisonAxis) -> str:
    if axis == "mesh_size_m":
        return f"outer radius = {value_m * 1.0e3:.3g} mm"
    return f"mesh size = {value_m * 1.0e6:.3g} µm"


def _plot_title(axis: ComparisonAxis) -> str:
    if axis == "mesh_size_m":
        return "Minimum positions over mesh refinement (coarse → fine)"
    return "Minimum positions over increasing outer radius"
