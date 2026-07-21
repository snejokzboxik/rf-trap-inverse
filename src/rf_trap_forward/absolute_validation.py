"""Small Data.txt check using absolute four-electrode displacements."""

from __future__ import annotations

import argparse
import csv
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .calibrated_validation import (
    CalibrationCase,
    CalibrationEvaluation,
    CalibrationRow,
    VoltageModel,
    default_geometry_variant,
    evaluate_calibration_case,
)
from .dataset import ReferenceDataset, load_reference_dataset

PREVIOUS_ROWS_1_10_MEAN_ERROR_M = 1.074555447769478e-3


@dataclass(frozen=True)
class AbsoluteDisplacementCheck:
    """The two requested robust rows 1--10 mapping evaluations."""

    evaluations: tuple[CalibrationEvaluation, ...]
    runtime_seconds: float


@dataclass(frozen=True)
class AbsoluteDisplacementOutputPaths:
    """Files written by the focused absolute-displacement check."""

    summary_csv: Path
    per_row_csv: Path
    report_markdown: Path


def run_absolute_displacement_check(
    dataset: ReferenceDataset,
    *,
    row_numbers: Sequence[int] = tuple(range(1, 11)),
    central_mesh_size_m: float = 500.0e-6,
) -> AbsoluteDisplacementCheck:
    """Run only identity and E1,E3,E2,E4 mappings with robust minima."""

    started = time.perf_counter()
    geometry = default_geometry_variant()
    voltage = VoltageModel("all-positive", (1.0, 1.0, 1.0, 1.0))
    cases = (
        CalibrationCase(
            name="absolute-identity",
            family="absolute-displacement-check",
            geometry=geometry,
            voltage=voltage,
            electrode_mapping=(1, 2, 3, 4),
        ),
        CalibrationCase(
            name="absolute-perm1324",
            family="absolute-displacement-check",
            geometry=geometry,
            voltage=voltage,
            electrode_mapping=(1, 3, 2, 4),
        ),
    )
    evaluations = tuple(
        evaluate_calibration_case(
            dataset,
            case,
            row_numbers,
            central_mesh_size_m=central_mesh_size_m,
            scope="absolute-displacement-rows1-10",
            maximum_parallel_rows=3,
            checkpoint_directory=None,
        )
        for case in cases
    )
    return AbsoluteDisplacementCheck(
        evaluations=evaluations,
        runtime_seconds=time.perf_counter() - started,
    )


def write_absolute_displacement_outputs(
    check: AbsoluteDisplacementCheck,
    output_directory: str | Path,
) -> AbsoluteDisplacementOutputPaths:
    """Write the two compact CSV tables and focused Markdown report."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = AbsoluteDisplacementOutputPaths(
        summary_csv=output / "summary.csv",
        per_row_csv=output / "per_row.csv",
        report_markdown=output / "report.md",
    )
    summary_records = [_summary_record(item) for item in check.evaluations]
    row_records = [
        _row_record(evaluation, row)
        for evaluation in check.evaluations
        for row in evaluation.rows
    ]
    _write_csv(paths.summary_csv, summary_records)
    _write_csv(paths.per_row_csv, row_records)
    paths.report_markdown.write_text(
        _markdown_report(check),
        encoding="utf-8",
    )
    return paths


def _summary_record(evaluation: CalibrationEvaluation) -> dict[str, object]:
    summary = evaluation.summary()
    improvement = (
        (PREVIOUS_ROWS_1_10_MEAN_ERROR_M - summary.mean_error_m)
        / PREVIOUS_ROWS_1_10_MEAN_ERROR_M
        if np.isfinite(summary.mean_error_m)
        else float("nan")
    )
    return {
        "mapping": evaluation.case.name,
        "electrode_permutation": "-".join(
            str(value) for value in evaluation.case.electrode_mapping
        ),
        "displacement_mode": "absolute-four-electrode",
        "central_mesh_size_m": evaluation.central_mesh_size_m,
        "central_mesh_size_um": 1.0e6 * evaluation.central_mesh_size_m,
        "selected_rows": summary.selected_rows,
        "completed_rows": summary.completed_rows,
        "exactly_three_rows": summary.exactly_three_rows,
        "matched_minima": summary.matched_minima,
        "mean_error_m": summary.mean_error_m,
        "mean_error_mm": 1.0e3 * summary.mean_error_m,
        "median_error_m": summary.median_error_m,
        "median_error_mm": 1.0e3 * summary.median_error_m,
        "maximum_error_m": summary.maximum_error_m,
        "maximum_error_mm": 1.0e3 * summary.maximum_error_m,
        "percentile_95_error_m": summary.percentile_95_error_m,
        "percentile_95_error_mm": 1.0e3 * summary.percentile_95_error_m,
        "selected_interpolation_sensitive": summary.selected_interpolation_sensitive,
        "rejected_candidates": summary.rejected_candidates,
        "summed_worker_runtime_seconds": summary.runtime_seconds,
        "relative_to_previous_rows1_10_mean_fraction": improvement,
        "validation_gate_passed": summary.validation_gate_passed,
    }


def _row_record(
    evaluation: CalibrationEvaluation,
    row: CalibrationRow,
) -> dict[str, object]:
    errors = row.errors_m()
    return {
        "mapping": evaluation.case.name,
        "electrode_permutation": "-".join(
            str(value) for value in evaluation.case.electrode_mapping
        ),
        "displacement_mode": "absolute-four-electrode",
        "row_number": row.row_number,
        "status": row.status,
        "completed": row.completed,
        "exactly_three_robust_minima": row.exactly_three_topology,
        "topology_candidate_count": row.topology_candidate_count,
        "node_count": row.node_count,
        "triangle_count": row.triangle_count,
        "relative_free_residual": row.relative_free_residual,
        "runtime_seconds": row.runtime_seconds,
        "mean_error_mm": _metric(errors, np.mean),
        "median_error_mm": _metric(errors, np.median),
        "maximum_error_mm": _metric(errors, np.max),
        "percentile_95_error_mm": _metric(
            errors,
            lambda values: np.percentile(values, 95.0),
        ),
        "selected_interpolation_sensitive": row.selected_interpolation_sensitive,
        "rejected_candidates": row.rejected_candidates,
        "error_type": row.error_type,
        "error_message": row.error_message,
    }


def _metric(
    values: np.ndarray,
    reducer: Callable[[np.ndarray], float | np.floating],
) -> float:
    if values.size == 0:
        return float("nan")
    return 1.0e3 * float(reducer(values))


def _write_csv(path: Path, records: list[dict[str, object]]) -> None:
    if not records:
        raise ValueError("cannot write an empty validation table")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _markdown_report(check: AbsoluteDisplacementCheck) -> str:
    lines = [
        "# Absolute four-electrode displacement check",
        "",
        "`Data.txt` rows are applied in the fixed outer-boundary frame: each "
        "electrode center equals its nominal center plus its raw displacement. "
        "Electrode 1 moves; the grounded outer circle remains centered at the origin.",
        "",
        "The check uses the real-scale all-positive geometry, 500 um local central "
        "mesh, robust minima mode, rows 1--10, and no geometry, voltage, or output "
        "calibration.",
        "",
        "| mapping | completed | exactly three | mean (mm) | median (mm) | max (mm) | p95 (mm) | change vs prior 1.07456 mm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for evaluation in check.evaluations:
        summary = evaluation.summary()
        change = (
            (PREVIOUS_ROWS_1_10_MEAN_ERROR_M - summary.mean_error_m)
            / PREVIOUS_ROWS_1_10_MEAN_ERROR_M
        )
        lines.append(
            f"| `{evaluation.case.name}` | {summary.completed_rows}/{summary.selected_rows} "
            f"| {summary.exactly_three_rows}/{summary.selected_rows} "
            f"| {_mm(summary.mean_error_m)} | {_mm(summary.median_error_m)} "
            f"| {_mm(summary.maximum_error_m)} | {_mm(summary.percentile_95_error_m)} "
            f"| {100.0 * change:+.3f}% |"
        )
    best = min(
        check.evaluations,
        key=lambda item: item.summary().mean_error_m,
    ).summary()
    improved = best.mean_error_m < PREVIOUS_ROWS_1_10_MEAN_ERROR_M
    lines.extend(
        (
            "",
            f"Focused wall time: {check.runtime_seconds:.3f} s.",
            "",
            "The best absolute-displacement result "
            + ("reduces" if improved else "does not reduce")
            + " the prior approximately 1.07 mm rows 1--10 mismatch. "
            "This is a convention correction only; the validation gate and physical-model "
            "interpretation must be judged from the metrics above.",
        )
    )
    return "\n".join(lines) + "\n"


def _mm(value_m: float) -> str:
    return "n/a" if not np.isfinite(value_m) else f"{1.0e3 * value_m:.6g}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rf-trap-absolute-displacement-check",
        description="Run the focused absolute-displacement Data.txt check.",
    )
    parser.add_argument("input", nargs="?", type=Path, default=Path("Data.txt"))
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("validation_results/absolute_displacement_check"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    dataset = load_reference_dataset(arguments.input)
    check = run_absolute_displacement_check(dataset)
    paths = write_absolute_displacement_outputs(check, arguments.output_directory)
    for evaluation in check.evaluations:
        summary = evaluation.summary()
        print(
            f"{evaluation.case.name}: mean={_mm(summary.mean_error_m)} mm, "
            f"median={_mm(summary.median_error_m)} mm, "
            f"max={_mm(summary.maximum_error_m)} mm, "
            f"p95={_mm(summary.percentile_95_error_m)} mm, "
            f"exactly_three={summary.exactly_three_rows}/{summary.selected_rows}"
        )
    print(f"report={paths.report_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
