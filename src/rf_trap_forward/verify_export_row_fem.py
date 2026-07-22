"""Verify that prediction-export rows reproduce their stored FEM minima.

This diagnostic deliberately reuses the practical synthetic-data worker.  It
does not alter the forward model or any source CSV: it reads one or more rows,
applies the documented Wolfram-to-FEM mapping, and compares robust minima with
the minima stored in the export.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from .absolute_validation import wolfram_to_fem_absolute_displacements_m
from .dataset import sort_points_by_polar_angle
from .geometry import build_geometry_from_absolute_displacements
from .minima_modes import RobustMinimaConfig
from .synthetic_dataset import (
    SyntheticSolveResult,
    _default_fem_worker,
    practical_generator_forward_config,
)


TRUE_W_COLUMNS = tuple(
    f"true_w{electrode}_{component}_m"
    for electrode in range(1, 5)
    for component in ("dx", "dy")
)
MINIMA_COLUMNS = tuple(
    f"min{minimum}_{component}_m"
    for minimum in range(1, 4)
    for component in ("x", "y")
)
FEM_COLUMNS = tuple(
    f"fem_f{electrode}_{component}_m"
    for electrode in range(1, 5)
    for component in ("dx", "dy")
)
SUMMARY_COLUMNS = (
    "row_index",
    "sample_id",
    "mapping",
    *TRUE_W_COLUMNS,
    *FEM_COLUMNS,
    *MINIMA_COLUMNS,
    *(f"recomputed_{column}" for column in MINIMA_COLUMNS),
    "min1_error_um",
    "min2_error_um",
    "min3_error_um",
    "mean_error_um",
    "max_error_um",
    "status",
    "error_type",
    "error_message",
    "accepted_candidate_count",
    "rejected_candidate_count",
    "total_candidate_count",
    "node_count",
    "triangle_count",
    "relative_free_residual",
    "runtime_seconds",
)


@dataclass(frozen=True)
class ExportRow:
    """One source row with its zero-based CSV index."""

    row_index: int
    sample_id: int
    true_wolfram_m: NDArray[np.float64]
    csv_minima_m: NDArray[np.float64]


@dataclass(frozen=True)
class MappingDiagnostic:
    """One FEM run for one row and one displacement mapping."""

    mapping: str
    fem_displacements_m: NDArray[np.float64]
    recomputed_minima_m: NDArray[np.float64]
    errors_um: NDArray[np.float64]
    status: str
    solve: SyntheticSolveResult
    error_type: str = ""
    error_message: str = ""


def load_export_rows(path: str | Path) -> tuple[ExportRow, ...]:
    """Load and validate true W displacements and stored minima from an export."""

    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"sample_id", *TRUE_W_COLUMNS, *MINIMA_COLUMNS}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required.difference(reader.fieldnames or ()))
            raise ValueError(f"export CSV is missing required columns: {missing}")
        rows: list[ExportRow] = []
        for row_index, row in enumerate(reader):
            try:
                sample_id = int(row["sample_id"])
                true_w = np.asarray(
                    [float(row[column]) for column in TRUE_W_COLUMNS], dtype=float
                )
                minima = np.asarray(
                    [float(row[column]) for column in MINIMA_COLUMNS], dtype=float
                ).reshape(3, 2)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"malformed numeric value at export row index {row_index}"
                ) from error
            if not np.all(np.isfinite(true_w)) or not np.all(np.isfinite(minima)):
                raise ValueError(f"non-finite numeric value at export row index {row_index}")
            rows.append(ExportRow(row_index, sample_id, true_w, minima))
    if not rows:
        raise ValueError("export CSV contains no rows")
    return tuple(rows)


def select_export_rows(
    rows: Sequence[ExportRow],
    *,
    row_indices: Sequence[int] = (),
    sample_ids: Sequence[int] = (),
) -> tuple[ExportRow, ...]:
    """Select rows by zero-based index or source sample ID."""

    if bool(row_indices) == bool(sample_ids):
        raise ValueError("provide row_indices or sample_ids, but not both")
    if row_indices:
        selected: list[ExportRow] = []
        for index in row_indices:
            if index < 0 or index >= len(rows):
                raise IndexError(f"row-index {index} is outside 0..{len(rows) - 1}")
            selected.append(rows[index])
        return tuple(selected)
    by_sample_id = {row.sample_id: row for row in rows}
    selected = []
    for sample_id in sample_ids:
        if sample_id not in by_sample_id:
            raise KeyError(f"sample-id {sample_id} is absent from the export")
        selected.append(by_sample_id[sample_id])
    return tuple(selected)


def wrong_direct_fem_mapping(true_wolfram_m: object) -> NDArray[np.float64]:
    """Return the intentionally wrong debugging mapping W1..W4 -> F1..F4."""

    array = np.asarray(true_wolfram_m, dtype=float)
    if array.shape == (8,):
        array = array.reshape(4, 2)
    if array.shape != (4, 2) or not np.all(np.isfinite(array)):
        raise ValueError("true_wolfram_m must have shape (4, 2) or (8,)")
    return array.copy()


def diagnose_row(
    row: ExportRow,
    *,
    config: object | None = None,
    robust_config: RobustMinimaConfig | None = None,
) -> tuple[MappingDiagnostic, MappingDiagnostic]:
    """Run canonical and wrong-direct mappings through the dataset worker."""

    forward_config = config or practical_generator_forward_config("practical")
    controls = robust_config or RobustMinimaConfig()
    canonical = wolfram_to_fem_absolute_displacements_m(row.true_wolfram_m).reshape(4, 2)
    wrong = wrong_direct_fem_mapping(row.true_wolfram_m)
    return (
        _run_mapping("canonical-[-W3,-W1,-W4,-W2]", canonical, row, forward_config, controls),
        _run_mapping("wrong-direct-W-as-F", wrong, row, forward_config, controls),
    )


def _run_mapping(
    name: str,
    fem_displacements_m: NDArray[np.float64],
    row: ExportRow,
    config: object,
    robust_config: RobustMinimaConfig,
) -> MappingDiagnostic:
    try:
        build_geometry_from_absolute_displacements(config.geometry, fem_displacements_m)
    except Exception as error:
        failed = SyntheticSolveResult.failure(type(error).__name__, str(error))
        return MappingDiagnostic(
            name,
            fem_displacements_m.copy(),
            np.empty((0, 2), dtype=float),
            np.full(3, np.nan),
            "geometry_failed",
            failed,
            type(error).__name__,
            str(error),
        )
    try:
        solve = _default_fem_worker(fem_displacements_m, config, robust_config)
    except Exception as error:
        solve = SyntheticSolveResult.failure(type(error).__name__, str(error))
    if solve.error_type:
        return MappingDiagnostic(
            name,
            fem_displacements_m.copy(),
            np.empty((0, 2), dtype=float),
            np.full(3, np.nan),
            "solver_failed",
            solve,
            solve.error_type,
            solve.error_message,
        )
    recomputed = (
        sort_points_by_polar_angle(solve.minima_positions_m)
        if solve.minima_positions_m.size
        else np.empty((0, 2), dtype=float)
    )
    if recomputed.shape != (3, 2) or solve.accepted_candidate_count != 3:
        return MappingDiagnostic(
            name,
            fem_displacements_m.copy(),
            recomputed,
            np.full(3, np.nan),
            "not_exactly_three",
            solve,
        )
    errors_um = 1.0e6 * np.linalg.norm(recomputed - row.csv_minima_m, axis=1)
    return MappingDiagnostic(
        name,
        fem_displacements_m.copy(),
        recomputed,
        errors_um,
        "ok",
        solve,
    )


def _finite_or_blank(value: float) -> float | str:
    return float(value) if math.isfinite(float(value)) else ""


def _summary_row(row: ExportRow, result: MappingDiagnostic) -> dict[str, object]:
    record: dict[str, object] = {
        "row_index": row.row_index,
        "sample_id": row.sample_id,
        "mapping": result.mapping,
        "status": result.status,
        "error_type": result.error_type,
        "error_message": result.error_message,
        "accepted_candidate_count": result.solve.accepted_candidate_count,
        "rejected_candidate_count": result.solve.rejected_candidate_count,
        "total_candidate_count": result.solve.total_candidate_count,
        "node_count": result.solve.node_count,
        "triangle_count": result.solve.triangle_count,
        "relative_free_residual": _finite_or_blank(result.solve.relative_free_residual),
        "runtime_seconds": result.solve.runtime_seconds,
        "mean_error_um": _finite_or_blank(np.nanmean(result.errors_um)),
        "max_error_um": _finite_or_blank(np.nanmax(result.errors_um)),
    }
    record.update(zip(TRUE_W_COLUMNS, row.true_wolfram_m, strict=True))
    record.update(zip(FEM_COLUMNS, result.fem_displacements_m.reshape(8), strict=True))
    record.update(zip(MINIMA_COLUMNS, row.csv_minima_m.reshape(6), strict=True))
    recomputed = (
        result.recomputed_minima_m.reshape(6)
        if result.recomputed_minima_m.shape == (3, 2)
        else np.full(6, np.nan)
    )
    record.update(
        zip(
            (f"recomputed_{column}" for column in MINIMA_COLUMNS),
            recomputed,
            strict=True,
        )
    )
    errors = result.errors_um if result.errors_um.shape == (3,) else np.full(3, np.nan)
    record.update(
        {f"min{index}_error_um": _finite_or_blank(errors[index - 1]) for index in range(1, 4)}
    )
    return record


def _write_row_report(
    path: Path,
    row: ExportRow,
    results: Sequence[MappingDiagnostic],
) -> None:
    lines = [
        f"# Export row FEM diagnostic: row index {row.row_index}",
        "",
        f"- Sample ID: `{row.sample_id}`",
        "- FEM configuration: practical synthetic-data configuration (500 µm central mesh)",
        "- Robust minima mode: yes",
        "- Canonical transform: `FEM = [-W3, -W1, -W4, -W2]`",
        "- Recomputed minima are sorted by `sort_points_by_polar_angle`.",
        "",
        "## Input displacements (metres)",
        "",
        "| electrode | dx | dy |",
        "|---|---:|---:|",
    ]
    for index, pair in enumerate(row.true_wolfram_m.reshape(4, 2), start=1):
        lines.append(f"| W{index} | {pair[0]:.12g} | {pair[1]:.12g} |")
    lines.extend(["", "## Stored minima versus recomputed minima", ""])
    lines.extend(
        [
            "| mapping | status | min1 error (µm) | min2 error (µm) | min3 error (µm) | mean (µm) | max (µm) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        errors = result.errors_um if result.errors_um.shape == (3,) else np.full(3, np.nan)
        values = ["" if not math.isfinite(float(value)) else f"{value:.6f}" for value in errors]
        mean = "" if not math.isfinite(float(np.nanmean(errors))) else f"{np.nanmean(errors):.6f}"
        maximum = "" if not math.isfinite(float(np.nanmax(errors))) else f"{np.nanmax(errors):.6f}"
        lines.append(
            f"| {result.mapping} | {result.status} | {values[0]} | {values[1]} | {values[2]} | {mean} | {maximum} |"
        )
    lines.extend(["", "## Stored minima (metres)", "", "```text", str(row.csv_minima_m), "```", ""])
    for result in results:
        lines.extend(
            [
                f"### {result.mapping}",
                "",
                f"FEM displacements (F1..F4):\n\n```text\n{result.fem_displacements_m}\n```",
                f"Recomputed minima (sorted):\n\n```text\n{result.recomputed_minima_m}\n```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def diagnose_export_rows(
    export_csv: str | Path,
    *,
    row_indices: Sequence[int] = (),
    sample_ids: Sequence[int] = (),
    output_dir: str | Path = "validation_results/export_row_fem_debug",
) -> tuple[Path, ...]:
    """Run selected rows and write per-row reports plus an aggregate CSV."""

    rows = select_export_rows(
        load_export_rows(export_csv), row_indices=row_indices, sample_ids=sample_ids
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    report_paths: list[Path] = []
    for row in rows:
        results = diagnose_row(row)
        summary_rows.extend(_summary_row(row, result) for result in results)
        report_path = output / f"row_index_{row.row_index}_report.md"
        _write_row_report(report_path, row, results)
        report_paths.append(report_path)
    summary_path = output / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(SUMMARY_COLUMNS))
        writer.writeheader()
        writer.writerows(summary_rows)
    return (summary_path, *report_paths)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rf-trap-verify-export-row-fem",
        description="Check prediction-export true displacements against robust FEM minima.",
    )
    parser.add_argument("--export-csv", type=Path, required=True)
    selectors = parser.add_mutually_exclusive_group(required=True)
    selectors.add_argument("--row-index", type=int, action="append")
    selectors.add_argument("--sample-id", type=int, action="append")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("validation_results/export_row_fem_debug")
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    paths = diagnose_export_rows(
        arguments.export_csv,
        row_indices=arguments.row_index or (),
        sample_ids=arguments.sample_id or (),
        output_dir=arguments.output_dir,
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
