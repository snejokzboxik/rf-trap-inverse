"""Milestone-5 real-scale validation and convention diagnostics."""

from __future__ import annotations

import argparse
import csv
import itertools
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dataset import ReferenceDataset, load_reference_dataset
from .real_scale import (
    DIAGONAL_ALTERNATING_POTENTIALS_V,
    REAL_COARSE_MESH_SIZES_M,
    REAL_INNER_RADIUS_M,
    real_scale_forward_config,
)
from .reference_validation import (
    ForwardRunner,
    ReferenceValidationReport,
    ReferenceValidationVariant,
    run_reference_validation,
    write_reference_validation_outputs,
)

MILESTONE_4_MEAN_ERROR_M = 3.18585e-3


@dataclass(frozen=True)
class ScaleValidationCase:
    """One mesh/convention case in the milestone-5 study."""

    name: str
    scope: str
    report: ReferenceValidationReport


@dataclass(frozen=True)
class ScaleValidationStudy:
    """All cases and timing for one real-scale validation study."""

    cases: tuple[ScaleValidationCase, ...]
    full_row_numbers: tuple[int, ...]
    runtime_seconds: float

    def best_full_case(self) -> ScaleValidationCase:
        """Return the best comparable full-row case by completion then error."""

        return select_best_case(self.cases, self.full_row_numbers)


@dataclass(frozen=True)
class ScaleValidationOutputPaths:
    """Artifacts written for the milestone-5 diagnostic study."""

    summary_csv: Path
    rows_csv: Path
    minima_csv: Path
    markdown_report: Path
    best_case_directory: Path
    plot_paths: tuple[Path, ...]


def select_best_case(
    cases: Iterable[ScaleValidationCase],
    required_rows: Sequence[int],
) -> ScaleValidationCase:
    """Select a comparable case, preferring completion before lower error."""

    required = tuple(int(value) for value in required_rows)
    comparable = [
        case
        for case in cases
        if tuple(row.row_number for row in case.report.rows) == required
    ]
    if not comparable:
        raise ValueError("no validation case covers the required rows")

    def rank(case: ScaleValidationCase) -> tuple[float, float, float]:
        summary = case.report.summary()
        mean = summary.mean_error_m
        maximum = summary.maximum_error_m
        return (
            -float(summary.completed_rows),
            mean if np.isfinite(mean) else float("inf"),
            maximum if np.isfinite(maximum) else float("inf"),
        )

    return min(comparable, key=rank)


def run_milestone_5_study(
    dataset: ReferenceDataset,
    *,
    row_numbers: Sequence[int] = tuple(range(1, 11)),
    mesh_sizes_m: Sequence[float] = REAL_COARSE_MESH_SIZES_M,
    screen_permutations: bool = True,
    runner: ForwardRunner | None = None,
) -> ScaleValidationStudy:
    """Run real-scale mesh, frame, polarity, and numbering diagnostics.

    All requested mesh sizes use the identity, electrode-1-relative,
    all-positive model on the full row set. Absolute-frame and alternating
    variants use the coarsest mesh. Non-identity E2--E4 permutations are first
    screened on up to three rows and promoted to the full row set only if the
    screen improves the identity result by at least two percent.
    """

    rows = tuple(int(value) for value in row_numbers)
    meshes = tuple(float(value) for value in mesh_sizes_m)
    if not rows or len(set(rows)) != len(rows):
        raise ValueError("row_numbers must be nonempty and unique")
    if not meshes or any(not np.isfinite(value) or value <= 0.0 for value in meshes):
        raise ValueError("mesh_sizes_m must contain finite positive values")

    started = time.perf_counter()
    cases: list[ScaleValidationCase] = []
    identity = (1, 2, 3, 4)
    relative_all_positive = _variant(
        "relative_all_positive_identity",
        "electrode1-relative",
        identity,
        "all-positive",
    )
    for mesh_size in meshes:
        cases.append(
            _run_case(
                dataset,
                rows,
                mesh_size,
                relative_all_positive,
                "full",
                runner,
            )
        )

    coarsest = meshes[0]
    cases.append(
        _run_case(
            dataset,
            rows,
            coarsest,
            _variant(
                "absolute_all_positive_identity",
                "absolute",
                identity,
                "all-positive",
            ),
            "full",
            runner,
        )
    )
    for mode in ("electrode1-relative", "absolute"):
        cases.append(
            _run_case(
                dataset,
                rows,
                coarsest,
                _variant(
                    f"{mode}_alternating_identity",
                    mode,
                    identity,
                    "alternating",
                ),
                "full",
                runner,
            )
        )

    if screen_permutations:
        screening_rows = rows[: min(3, len(rows))]
        screen_cases = []
        for tail in itertools.permutations((2, 3, 4)):
            permutation = (1, *tail)
            if permutation == identity:
                continue
            case = _run_case(
                dataset,
                screening_rows,
                coarsest,
                _variant(
                    "relative_all_positive_perm_" + "".join(map(str, permutation)),
                    "electrode1-relative",
                    permutation,
                    "all-positive",
                ),
                "screen",
                runner,
            )
            cases.append(case)
            screen_cases.append(case)

        identity_case = cases[0]
        identity_screen_mean = _mean_for_rows(identity_case.report, screening_rows)
        promoted = _best_screen_if_materially_better(
            screen_cases,
            identity_screen_mean,
            len(screening_rows),
        )
        if promoted is not None:
            cases.append(
                _run_case(
                    dataset,
                    rows,
                    coarsest,
                    promoted.report.variant,
                    "full-promoted",
                    runner,
                )
            )

    return ScaleValidationStudy(
        cases=tuple(cases),
        full_row_numbers=rows,
        runtime_seconds=time.perf_counter() - started,
    )


def write_milestone_5_outputs(
    study: ScaleValidationStudy,
    output_directory: str | Path,
) -> ScaleValidationOutputPaths:
    """Write all-case CSVs, report, and detailed plots for the best full case."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    best = study.best_full_case()
    best_directory = output / "best_case"
    best_paths = write_reference_validation_outputs(best.report, best_directory)
    paths = ScaleValidationOutputPaths(
        summary_csv=output / "variant_summary.csv",
        rows_csv=output / "validation_rows.csv",
        minima_csv=output / "validation_minima.csv",
        markdown_report=output / "milestone_5_report.md",
        best_case_directory=best_directory,
        plot_paths=best_paths.plot_paths,
    )
    _write_csv(paths.summary_csv, _case_summary_records(study))
    _write_csv(paths.rows_csv, _row_records(study))
    _write_csv(paths.minima_csv, _minimum_records(study))
    paths.markdown_report.write_text(_markdown_report(study), encoding="utf-8")
    return paths


def _variant(
    name: str,
    displacement_mode: str,
    permutation: tuple[int, int, int, int],
    polarity: str,
) -> ReferenceValidationVariant:
    return ReferenceValidationVariant(
        name=name,
        displacement_mode=displacement_mode,  # type: ignore[arg-type]
        electrode_permutation=permutation,
        polarity_name=polarity,
    )


def _run_case(
    dataset: ReferenceDataset,
    rows: Sequence[int],
    mesh_size_m: float,
    variant: ReferenceValidationVariant,
    scope: str,
    runner: ForwardRunner | None,
) -> ScaleValidationCase:
    potentials = (
        DIAGONAL_ALTERNATING_POTENTIALS_V
        if variant.polarity_name == "alternating"
        else None
    )
    config = real_scale_forward_config(
        mesh_size_m=mesh_size_m,
        electrode_potentials_v=potentials,
    )
    report = run_reference_validation(
        dataset,
        config,
        rows,
        runner=runner,
        variant=variant,
    )
    mesh_label = f"h{mesh_size_m * 1.0e3:g}mm"
    return ScaleValidationCase(
        name=f"{variant.name}_{mesh_label}",
        scope=scope,
        report=report,
    )


def _mean_for_rows(
    report: ReferenceValidationReport,
    row_numbers: Sequence[int],
) -> float:
    selected = set(row_numbers)
    errors = [
        match.distance_m
        for row in report.rows
        if row.row_number in selected
        for match in row.matches
    ]
    return float(np.mean(errors)) if errors else float("nan")


def _best_screen_if_materially_better(
    cases: Sequence[ScaleValidationCase],
    identity_mean_m: float,
    expected_completed_rows: int,
) -> ScaleValidationCase | None:
    candidates = [
        case
        for case in cases
        if case.report.summary().completed_rows == expected_completed_rows
        and np.isfinite(case.report.summary().mean_error_m)
    ]
    if not candidates or not np.isfinite(identity_mean_m):
        return None
    best = min(candidates, key=lambda case: case.report.summary().mean_error_m)
    if best.report.summary().mean_error_m < 0.98 * identity_mean_m:
        return best
    return None


def _case_summary_records(study: ScaleValidationStudy) -> list[dict[str, object]]:
    records = []
    for case in study.cases:
        report = case.report
        summary = report.summary()
        geometry = report.model_config.geometry
        records.append(
            {
                "case": case.name,
                "scope": case.scope,
                "rows": ",".join(str(row.row_number) for row in report.rows),
                "mesh_size_m": report.model_config.mesh.characteristic_length_m,
                "outer_radius_m": geometry.outer_radius_m,
                "electrode_radius_m": geometry.electrode_radius_m,
                "electrode_center_radius_m": float(
                    np.linalg.norm(geometry.nominal_centers_m[0])
                ),
                "inner_radius_m": float(
                    np.linalg.norm(geometry.nominal_centers_m[0])
                    - geometry.electrode_radius_m
                ),
                "search_half_width_m": report.model_config.minima.search_half_extent_m,
                "displacement_mode": report.variant.displacement_mode,
                "electrode_permutation": "-".join(
                    map(str, report.variant.electrode_permutation)
                ),
                "polarity": report.variant.polarity_name,
                "electrode_potentials_v": ",".join(
                    f"{value:g}" for value in geometry.resolved_electrode_potentials_v
                ),
                "selected_rows": summary.selected_rows,
                "completed_rows": summary.completed_rows,
                "failed_rows": summary.failed_rows,
                "rows_exactly_three_physical_minima": (
                    summary.rows_with_exactly_three_physical_minima
                ),
                "matched_minima": summary.matched_minima,
                "mean_error_um": summary.mean_error_m * 1.0e6,
                "median_error_um": summary.median_error_m * 1.0e6,
                "maximum_error_um": summary.maximum_error_m * 1.0e6,
                "percentile_95_error_um": summary.percentile_95_error_m * 1.0e6,
                "runtime_seconds": report.runtime_seconds,
            }
        )
    return records


def _row_records(study: ScaleValidationStudy) -> list[dict[str, object]]:
    records = []
    for case in study.cases:
        for row in case.report.rows:
            errors = row.error_distances_m()
            observation = row.observation
            records.append(
                {
                    "case": case.name,
                    "scope": case.scope,
                    "row_number": row.row_number,
                    "status": row.status,
                    "runtime_seconds": row.runtime_seconds,
                    "computed_minimum_count": (
                        0 if observation is None else len(observation.minima_positions_m)
                    ),
                    "hessian_validated_candidates": (
                        "" if observation is None else observation.hessian_validated_candidates
                    ),
                    "node_count": "" if observation is None else observation.node_count,
                    "triangle_count": (
                        "" if observation is None else observation.triangle_count
                    ),
                    "relative_free_residual": (
                        "" if observation is None else observation.relative_free_residual
                    ),
                    "mean_error_um": (
                        "" if not errors.size else float(np.mean(errors) * 1.0e6)
                    ),
                    "median_error_um": (
                        "" if not errors.size else float(np.median(errors) * 1.0e6)
                    ),
                    "maximum_error_um": (
                        "" if not errors.size else float(np.max(errors) * 1.0e6)
                    ),
                    "error_type": row.error_type,
                    "error_message": row.error_message,
                }
            )
    return records


def _minimum_records(study: ScaleValidationStudy) -> list[dict[str, object]]:
    records = []
    for case in study.cases:
        for row in case.report.rows:
            for match in row.matches:
                records.append(
                    {
                        "case": case.name,
                        "scope": case.scope,
                        "row_number": row.row_number,
                        "reference_index": match.reference_index,
                        "computed_index": match.computed_index,
                        "reference_comparison_x_m": match.reference_position_m[0],
                        "reference_comparison_y_m": match.reference_position_m[1],
                        "computed_comparison_x_m": match.computed_position_m[0],
                        "computed_comparison_y_m": match.computed_position_m[1],
                        "delta_x_m": match.delta_m[0],
                        "delta_y_m": match.delta_m[1],
                        "error_m": match.distance_m,
                        "error_um": match.distance_m * 1.0e6,
                        "error_mm": match.distance_m * 1.0e3,
                    }
                )
    return records


def _write_csv(path: Path, records: list[dict[str, object]]) -> None:
    if not records:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _markdown_report(study: ScaleValidationStudy) -> str:
    best = study.best_full_case()
    best_summary = best.report.summary()
    config = best.report.model_config
    geometry = config.geometry
    center_radius = float(np.linalg.norm(geometry.nominal_centers_m[0]))
    improvement = 1.0 - best_summary.mean_error_m / MILESTONE_4_MEAN_ERROR_M
    safe = _safe_for_dataset_generation(best.report)
    identity_cases = [
        case
        for case in study.cases
        if case.scope == "full"
        and case.report.variant.name == "relative_all_positive_identity"
    ]
    identity_coarse = identity_cases[0].report.summary()
    identity_fine = identity_cases[-1].report.summary()
    best_observations = [
        row.observation for row in best.report.rows if row.observation is not None
    ]
    mesh_diagnostic = (
        f"Its FEM meshes contain {min(obs.node_count for obs in best_observations)}"
        f"--{max(obs.node_count for obs in best_observations)} nodes and "
        f"{min(obs.triangle_count for obs in best_observations)}--"
        f"{max(obs.triangle_count for obs in best_observations)} triangles; "
        f"the maximum relative free residual is "
        f"`{max(obs.relative_free_residual for obs in best_observations):.6g}`."
        if best_observations
        else "No best-case mesh diagnostics are available because every row failed."
    )
    alternating_failures = [
        row.error_message
        for case in study.cases
        if case.report.variant.polarity_name == "alternating"
        for row in case.report.rows
        if row.error_message
    ]
    alternating_counts = sorted(
        {
            int(match.group(1))
            for message in alternating_failures
            if (match := re.search(r"found (\d+) validated minima", message))
        }
    )
    lines = [
        "# Milestone 5: real-scale FEM/reference validation",
        "",
        "## Geometry and numerical setup",
        "",
        f"- Outer-boundary radius: `{geometry.outer_radius_m * 1.0e3:.6g} mm`",
        f"- Electrode radius: `{geometry.electrode_radius_m * 1.0e3:.6g} mm`",
        f"- Inner radius (centre to nearest surface): `{REAL_INNER_RADIUS_M * 1.0e3:.6g} mm`",
        f"- Electrode-centre radius: `{center_radius * 1.0e3:.6g} mm`",
        f"- Diagonal coordinate `a`: `{abs(geometry.nominal_centers_m[0][0]) * 1.0e3:.9g} mm`",
        "- Numbering: `E1=(-a,+a), E2=(+a,+a), E3=(-a,-a), E4=(+a,-a)`.",
        f"- Search square: `±{config.minima.search_half_extent_m * 1.0e3:.6g} mm` in x and y.",
        "- Full-row coarse mesh sizes: `"
        + ", ".join(
            f"{case.report.model_config.mesh.characteristic_length_m * 1.0e3:g} mm"
            for case in study.cases
            if case.scope == "full"
            and case.report.variant.name == "relative_all_positive_identity"
        )
        + "`.",
        "- The alternating diagnostic uses checkerboard potentials",
        "  `(+1, -1, -1, +1) V` in E1--E4 order. It is not a default change.",
        "",
        "## Case summary",
        "",
        "| case | scope | rows ok | exact-three rows | mean (mm) | median (mm) | max (mm) | p95 (mm) | runtime (s) |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in study.cases:
        summary = case.report.summary()
        lines.append(
            f"| {case.name} | {case.scope} "
            f"| {summary.completed_rows}/{summary.selected_rows} "
            f"| {summary.rows_with_exactly_three_physical_minima} "
            f"| {_mm(summary.mean_error_m)} | {_mm(summary.median_error_m)} "
            f"| {_mm(summary.maximum_error_m)} "
            f"| {_mm(summary.percentile_95_error_m)} "
            f"| {case.report.runtime_seconds:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Diagnosis",
            "",
            "For identity numbering, refinement from "
            f"{identity_cases[0].report.model_config.mesh.characteristic_length_m * 1.0e3:g} "
            "to "
            f"{identity_cases[-1].report.model_config.mesh.characteristic_length_m * 1.0e3:g} mm "
            f"changes mean error from {_mm(identity_coarse.mean_error_m)} to "
            f"{_mm(identity_fine.mean_error_m)} mm. Exactly-three topology improves "
            f"from {identity_coarse.rows_with_exactly_three_physical_minima}/10 to "
            f"{identity_fine.rows_with_exactly_three_physical_minima}/10, so it is "
            "not stable across all rows/refinements and refinement alone does not "
            "remove the mismatch.",
            "",
            f"The best comparable full-row case is `{best.name}` with "
            f"{best_summary.completed_rows}/{best_summary.selected_rows} completed rows, "
            f"mean error `{_mm(best_summary.mean_error_m)} mm`, median "
            f"`{_mm(best_summary.median_error_m)} mm`, maximum "
            f"`{_mm(best_summary.maximum_error_m)} mm`, and p95 "
            f"`{_mm(best_summary.percentile_95_error_m)} mm`.",
            mesh_diagnostic,
            "",
            f"Against the Milestone-4 mean of `3.18585 mm`, this is a "
            f"`{improvement * 100.0:.3f}%` reduction. The full CSV tables retain "
            "failed rows; no failed validation is hidden.",
            "",
            "The absolute and electrode-1-relative cases explicitly test coordinate",
            "origin handling. Permutation cases change only the source-to-FEM mapping",
            "for E2--E4. The alternating-polarity cases test a clean diagnostic",
            "Dirichlet variant without changing the all-positive default.",
            f"The best map is FEM E1--E4 <- source E"
            + ",E".join(map(str, best.report.variant.electrode_permutation))
            + "; it is a diagnostic hypothesis, not a proven dataset convention.",
            "",
            (
                "Alternating-polarity failures reported validated-minimum counts "
                f"{alternating_counts}; this topology does not supply the required "
                "three minima in the tested four-electrode model."
                if alternating_counts
                else "Alternating-polarity topology results are retained in the CSV files."
            ),
            "",
            "The reference article concerns an eight-rod octupole. Real-scale",
            "dimensions can improve coordinate scale agreement, but do not establish",
            "physical equivalence of the present four-electrode boundary-value model.",
            "",
            "## Dataset-generation decision",
            "",
            (
                "**PASS:** the conservative validation gate is met."
                if safe
                else "**NOT SAFE YET:** the conservative validation gate is not met."
            ),
            "The gate requires all ten rows to complete with exactly three physical",
            "minima, mean error at most 0.25 mm, and maximum error at most 0.5 mm.",
            "This gate is an explicit project decision criterion, not a fitted model",
            "parameter. No ML or synthetic dataset generation was performed.",
            "",
            "Detailed CSVs and the best-case per-row plots are stored beside this",
            "report under `best_case/`.",
            "",
            f"Total diagnostic runtime: `{study.runtime_seconds:.3f} s`.",
            "",
        ]
    )
    return "\n".join(lines)


def _safe_for_dataset_generation(report: ReferenceValidationReport) -> bool:
    summary = report.summary()
    return bool(
        summary.selected_rows == 10
        and summary.completed_rows == 10
        and summary.rows_with_exactly_three_physical_minima == 10
        and summary.mean_error_m <= 0.25e-3
        and summary.maximum_error_m <= 0.50e-3
    )


def _mm(value_m: float) -> str:
    return "n/a" if not np.isfinite(value_m) else f"{value_m * 1.0e3:.6g}"


def build_parser() -> argparse.ArgumentParser:
    """Build the milestone-5 diagnostic command-line parser."""

    parser = argparse.ArgumentParser(
        prog="rf-trap-scale-validation",
        description="Run real-scale mesh and convention diagnostics on Data.txt.",
    )
    parser.add_argument("input", nargs="?", type=Path, default=Path("Data.txt"))
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("validation_results") / "milestone_5",
    )
    parser.add_argument(
        "--mesh-sizes-mm",
        default="2.0,1.5,1.0",
        help="Comma-separated full-row identity mesh sizes in millimetres.",
    )
    parser.add_argument("--skip-permutation-screen", action="store_true")
    return parser


def _parse_mesh_sizes_mm(value: str) -> tuple[float, ...]:
    try:
        sizes = tuple(float(part.strip()) * 1.0e-3 for part in value.split(","))
    except ValueError as error:
        raise ValueError("mesh sizes must be comma-separated numbers") from error
    if not sizes:
        raise ValueError("at least one mesh size is required")
    return sizes


def main(argv: Sequence[str] | None = None) -> int:
    """Run the milestone-5 study and print its principal result paths."""

    arguments = build_parser().parse_args(argv)
    dataset = load_reference_dataset(arguments.input)
    try:
        mesh_sizes = _parse_mesh_sizes_mm(arguments.mesh_sizes_mm)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    study = run_milestone_5_study(
        dataset,
        mesh_sizes_m=mesh_sizes,
        screen_permutations=not arguments.skip_permutation_screen,
    )
    paths = write_milestone_5_outputs(study, arguments.output_directory)
    best = study.best_full_case()
    summary = best.report.summary()
    print(f"best case: {best.name}")
    print(f"completed rows: {summary.completed_rows}/{summary.selected_rows}")
    print(f"mean error: {_mm(summary.mean_error_m)} mm")
    print(f"runtime: {study.runtime_seconds:.3f} s")
    print(f"summary CSV: {paths.summary_csv}")
    print(f"row CSV: {paths.rows_csv}")
    print(f"minimum CSV: {paths.minima_csv}")
    print(f"report: {paths.markdown_report}")
    print(f"plots: {len(paths.plot_paths)} under {paths.best_case_directory / 'plots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
