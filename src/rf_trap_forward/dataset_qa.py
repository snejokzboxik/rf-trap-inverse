"""Focused integrity and distribution audit for generated FEM datasets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import build_geometry_from_absolute_displacements
from .real_scale import REAL_SEARCH_HALF_WIDTH_M, real_scale_geometry_config
from .synthetic_dataset import (
    CLEAN_CSV_COLUMNS,
    DEFAULT_AMBIGUOUS_MINIMUM_DISTANCE_M,
    REJECTED_CSV_COLUMNS,
    minimum_pairwise_distance_m,
)

SUMMARY_STAT_COLUMNS = (
    "quantity",
    "component",
    "count",
    "unit",
    "minimum",
    "maximum",
    "mean",
    "std",
    "median",
)
QA_MAX_DISPLACEMENT_M = 500.0e-6


@dataclass(frozen=True)
class SummaryStatistic:
    """One finite distribution summary in SI units."""

    quantity: str
    component: str
    count: int
    unit: str
    minimum: float
    maximum: float
    mean: float
    std: float
    median: float


@dataclass(frozen=True)
class DatasetQAAudit:
    """Complete integrity checks, numerical arrays, and distribution summaries."""

    clean_schema: tuple[str, ...]
    rejected_schema: tuple[str, ...]
    summary: dict[str, object]
    clean_row_count: int
    rejected_row_count: int
    valid_clean_row_count: int
    sample_ids: NDArray[np.int64]
    raw_wolfram_displacements_m: NDArray[np.float64]
    fem_displacements_m: NDArray[np.float64]
    minima_positions_m: NDArray[np.float64]
    reported_pairwise_distances_m: NDArray[np.float64]
    rejected_candidate_counts: NDArray[np.int64]
    malformed_numeric_cells: tuple[str, ...]
    nonfinite_numeric_cells: tuple[str, ...]
    duplicate_full_rows: int
    duplicate_sample_ids: int
    duplicate_wolfram_inputs: int
    duplicate_minima_outputs: int
    displacement_bound_violations: int
    transform_violations: int
    search_region_violations: int
    vacuum_domain_violations: int
    invalid_geometries: int
    pairwise_threshold_violations: int
    pairwise_value_mismatches: int
    polar_order_violations: int
    minimum_electrode_clearance_m: float
    minimum_outer_clearance_m: float
    maximum_absolute_minimum_coordinate_m: float
    maximum_pairwise_value_error_m: float
    statistics: tuple[SummaryStatistic, ...]
    critical_issues: tuple[str, ...]
    observations: tuple[str, ...]

    @property
    def ml_ready(self) -> bool:
        """Return whether all integrity and geometry gates pass."""

        return not self.critical_issues


@dataclass(frozen=True)
class DatasetQAOutputPaths:
    """Markdown, CSV, and plot paths produced by one audit."""

    report_markdown: Path
    summary_stats_csv: Path
    plots_directory: Path
    displacement_histogram_png: Path
    minima_scatter_png: Path
    minima_histogram_png: Path
    pairwise_histogram_png: Path


@dataclass(frozen=True)
class _ParsedCleanRows:
    sample_ids: NDArray[np.int64]
    seeds: NDArray[np.int64]
    statuses: tuple[str, ...]
    raw_wolfram_displacements_m: NDArray[np.float64]
    fem_displacements_m: NDArray[np.float64]
    minima_positions_m: NDArray[np.float64]
    pairwise_distances_m: NDArray[np.float64]
    rejected_candidate_counts: NDArray[np.int64]
    malformed_cells: tuple[str, ...]
    nonfinite_cells: tuple[str, ...]


def count_polar_order_violations(minima_positions_m: ArrayLike) -> int:
    """Count samples whose three minima are not monotone in ``[0, 2*pi)``."""

    minima = np.asarray(minima_positions_m, dtype=float)
    if minima.ndim != 3 or minima.shape[1:] != (3, 2):
        raise ValueError("minima_positions_m must have shape (n, 3, 2)")
    if not np.all(np.isfinite(minima)):
        raise ValueError("minima_positions_m must be finite")
    angles = np.mod(np.arctan2(minima[:, :, 1], minima[:, :, 0]), 2.0 * np.pi)
    return int(np.count_nonzero(np.any(np.diff(angles, axis=1) < -1.0e-12, axis=1)))


def audit_generated_dataset(
    clean_csv: str | Path,
    rejected_csv: str | Path,
    summary_json: str | Path,
) -> DatasetQAAudit:
    """Audit one generated clean/rejected split without changing source files."""

    clean_schema, clean_rows = _read_csv(Path(clean_csv))
    rejected_schema, rejected_rows = _read_csv(Path(rejected_csv))
    summary, summary_errors = _read_summary(Path(summary_json))
    parsed = _parse_clean_rows(clean_rows)
    rejected_malformed, rejected_nonfinite = _scan_rejected_numeric_cells(
        rejected_rows
    )
    malformed = parsed.malformed_cells + rejected_malformed
    nonfinite = parsed.nonfinite_cells + rejected_nonfinite + tuple(summary_errors)

    duplicate_full = _duplicate_full_rows(clean_rows, clean_schema)
    duplicate_ids = _duplicate_scalar_count(parsed.sample_ids)
    duplicate_inputs = _duplicate_array_rows(parsed.raw_wolfram_displacements_m)
    duplicate_outputs = _duplicate_array_rows(parsed.minima_positions_m)

    expected_limit = QA_MAX_DISPLACEMENT_M
    bound_violations = int(
        np.count_nonzero(
            np.abs(parsed.raw_wolfram_displacements_m) > expected_limit + 1.0e-15
        )
        + np.count_nonzero(
            np.abs(parsed.fem_displacements_m) > expected_limit + 1.0e-15
        )
    )
    expected_fem = -parsed.raw_wolfram_displacements_m[:, (2, 0, 3, 1), :]
    transform_errors = np.max(
        np.abs(parsed.fem_displacements_m - expected_fem),
        axis=(1, 2),
        initial=0.0,
    )
    transform_violations = int(np.count_nonzero(transform_errors > 1.0e-15))

    maximum_absolute_minimum = _finite_max(
        np.abs(parsed.minima_positions_m),
    )
    search_violations = int(
        np.count_nonzero(
            np.any(
                np.abs(parsed.minima_positions_m) > REAL_SEARCH_HALF_WIDTH_M + 1.0e-12,
                axis=2,
            )
        )
    )
    domain = _domain_checks(
        parsed.fem_displacements_m,
        parsed.minima_positions_m,
    )

    recomputed_pairwise = np.asarray(
        [minimum_pairwise_distance_m(points) for points in parsed.minima_positions_m],
        dtype=float,
    )
    pairwise_errors = np.abs(
        recomputed_pairwise - parsed.pairwise_distances_m
    )
    maximum_pairwise_error = _finite_max(pairwise_errors)
    pairwise_mismatches = int(np.count_nonzero(pairwise_errors > 1.0e-15))
    pairwise_limit = float(
        summary.get(
            "ambiguous_minimum_distance_m",
            DEFAULT_AMBIGUOUS_MINIMUM_DISTANCE_M,
        )
    )
    pairwise_violations = int(
        np.count_nonzero(parsed.pairwise_distances_m < pairwise_limit - 1.0e-15)
    )
    polar_violations = (
        count_polar_order_violations(parsed.minima_positions_m)
        if parsed.minima_positions_m.size
        else 0
    )
    statistics = _build_statistics(
        parsed.raw_wolfram_displacements_m,
        parsed.fem_displacements_m,
        parsed.minima_positions_m,
        parsed.pairwise_distances_m,
    )
    critical = _critical_issues(
        clean_schema=clean_schema,
        rejected_schema=rejected_schema,
        clean_rows=len(clean_rows),
        rejected_rows=len(rejected_rows),
        valid_rows=parsed.sample_ids.size,
        summary=summary,
        malformed=malformed,
        nonfinite=nonfinite,
        duplicate_full=duplicate_full,
        duplicate_ids=duplicate_ids,
        duplicate_inputs=duplicate_inputs,
        bound_violations=bound_violations,
        transform_violations=transform_violations,
        search_violations=search_violations,
        domain_violations=domain[0],
        invalid_geometries=domain[1],
        pairwise_violations=pairwise_violations,
        pairwise_mismatches=pairwise_mismatches,
        polar_violations=polar_violations,
        statuses=parsed.statuses,
        sample_ids=parsed.sample_ids,
    )
    observations = _distribution_observations(
        summary,
        parsed.raw_wolfram_displacements_m,
        parsed.minima_positions_m,
        parsed.pairwise_distances_m,
        parsed.rejected_candidate_counts,
        parsed.sample_ids,
        len(rejected_rows),
    )
    return DatasetQAAudit(
        clean_schema=clean_schema,
        rejected_schema=rejected_schema,
        summary=summary,
        clean_row_count=len(clean_rows),
        rejected_row_count=len(rejected_rows),
        valid_clean_row_count=int(parsed.sample_ids.size),
        sample_ids=parsed.sample_ids,
        raw_wolfram_displacements_m=parsed.raw_wolfram_displacements_m,
        fem_displacements_m=parsed.fem_displacements_m,
        minima_positions_m=parsed.minima_positions_m,
        reported_pairwise_distances_m=parsed.pairwise_distances_m,
        rejected_candidate_counts=parsed.rejected_candidate_counts,
        malformed_numeric_cells=malformed,
        nonfinite_numeric_cells=nonfinite,
        duplicate_full_rows=duplicate_full,
        duplicate_sample_ids=duplicate_ids,
        duplicate_wolfram_inputs=duplicate_inputs,
        duplicate_minima_outputs=duplicate_outputs,
        displacement_bound_violations=bound_violations,
        transform_violations=transform_violations,
        search_region_violations=search_violations,
        vacuum_domain_violations=domain[0],
        invalid_geometries=domain[1],
        pairwise_threshold_violations=pairwise_violations,
        pairwise_value_mismatches=pairwise_mismatches,
        polar_order_violations=polar_violations,
        minimum_electrode_clearance_m=domain[2],
        minimum_outer_clearance_m=domain[3],
        maximum_absolute_minimum_coordinate_m=maximum_absolute_minimum,
        maximum_pairwise_value_error_m=maximum_pairwise_error,
        statistics=statistics,
        critical_issues=critical,
        observations=observations,
    )


def write_dataset_qa_outputs(
    audit: DatasetQAAudit,
    output_directory: str | Path,
) -> DatasetQAOutputPaths:
    """Write the requested Markdown report, statistics CSV, and plots."""

    output = Path(output_directory)
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    paths = DatasetQAOutputPaths(
        report_markdown=output / "qa_report.md",
        summary_stats_csv=output / "summary_stats.csv",
        plots_directory=plots,
        displacement_histogram_png=plots / "displacement_coordinates.png",
        minima_scatter_png=plots / "minima_positions.png",
        minima_histogram_png=plots / "minima_coordinates.png",
        pairwise_histogram_png=plots / "minimum_pairwise_distance.png",
    )
    _write_statistics_csv(paths.summary_stats_csv, audit.statistics)
    _write_displacement_histogram(audit, paths.displacement_histogram_png)
    _write_minima_scatter(audit, paths.minima_scatter_png)
    _write_minima_histogram(audit, paths.minima_histogram_png)
    _write_pairwise_histogram(audit, paths.pairwise_histogram_png)
    paths.report_markdown.write_text(
        _markdown_report(audit),
        encoding="utf-8",
    )
    return paths


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        schema = tuple(reader.fieldnames or ())
        return schema, list(reader)


def _read_summary(path: Path) -> tuple[dict[str, object], list[str]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        return {}, [f"summary_json:{type(error).__name__}:{error}"]
    if not isinstance(value, dict):
        return {}, ["summary_json:root is not an object"]
    errors = []
    _collect_nonfinite_json(value, "summary_json", errors)
    return value, errors


def _collect_nonfinite_json(value: object, path: str, errors: list[str]) -> None:
    if isinstance(value, float) and not np.isfinite(value):
        errors.append(path)
    elif isinstance(value, dict):
        for key, item in value.items():
            _collect_nonfinite_json(item, f"{path}.{key}", errors)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _collect_nonfinite_json(item, f"{path}[{index}]", errors)


def _parse_clean_rows(rows: Sequence[dict[str, str]]) -> _ParsedCleanRows:
    sample_ids = []
    seeds = []
    statuses = []
    raw = []
    fem = []
    minima = []
    pairwise = []
    rejected_counts = []
    malformed = []
    nonfinite = []
    for row_index, row in enumerate(rows, start=2):
        try:
            sample_id = _parse_integer(row, "sample_id", row_index, malformed)
            seed = _parse_integer(row, "seed", row_index, malformed)
            raw_row = np.asarray(
                [
                    [
                        _parse_number(row, f"w{i}_dx_m", row_index, malformed, nonfinite),
                        _parse_number(row, f"w{i}_dy_m", row_index, malformed, nonfinite),
                    ]
                    for i in range(1, 5)
                ]
            )
            fem_row = np.asarray(
                [
                    [
                        _parse_number(row, f"f{i}_dx_m", row_index, malformed, nonfinite),
                        _parse_number(row, f"f{i}_dy_m", row_index, malformed, nonfinite),
                    ]
                    for i in range(1, 5)
                ]
            )
            minima_row = np.asarray(
                [
                    [
                        _parse_number(row, f"min{i}_x_m", row_index, malformed, nonfinite),
                        _parse_number(row, f"min{i}_y_m", row_index, malformed, nonfinite),
                    ]
                    for i in range(1, 4)
                ]
            )
            pairwise_value = _parse_number(
                row,
                "min_pairwise_distance_m",
                row_index,
                malformed,
                nonfinite,
            )
            rejected_count = _parse_integer(
                row,
                "rejected_candidate_count",
                row_index,
                malformed,
            )
            if not np.all(np.isfinite(raw_row)) or not np.all(np.isfinite(fem_row)):
                raise ValueError("nonfinite displacement")
            if not np.all(np.isfinite(minima_row)) or not np.isfinite(pairwise_value):
                raise ValueError("nonfinite output")
        except (KeyError, ValueError):
            continue
        sample_ids.append(sample_id)
        seeds.append(seed)
        statuses.append(row.get("status", ""))
        raw.append(raw_row)
        fem.append(fem_row)
        minima.append(minima_row)
        pairwise.append(pairwise_value)
        rejected_counts.append(rejected_count)
    return _ParsedCleanRows(
        sample_ids=np.asarray(sample_ids, dtype=np.int64),
        seeds=np.asarray(seeds, dtype=np.int64),
        statuses=tuple(statuses),
        raw_wolfram_displacements_m=_stack_or_empty(raw, (4, 2)),
        fem_displacements_m=_stack_or_empty(fem, (4, 2)),
        minima_positions_m=_stack_or_empty(minima, (3, 2)),
        pairwise_distances_m=np.asarray(pairwise, dtype=float),
        rejected_candidate_counts=np.asarray(rejected_counts, dtype=np.int64),
        malformed_cells=tuple(malformed),
        nonfinite_cells=tuple(nonfinite),
    )


def _parse_number(
    row: dict[str, str],
    column: str,
    row_index: int,
    malformed: list[str],
    nonfinite: list[str],
) -> float:
    location = f"clean:{row_index}:{column}"
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError):
        malformed.append(location)
        raise ValueError(location) from None
    if not np.isfinite(value):
        nonfinite.append(location)
        raise ValueError(location)
    return value


def _parse_integer(
    row: dict[str, str],
    column: str,
    row_index: int,
    malformed: list[str],
) -> int:
    location = f"clean:{row_index}:{column}"
    try:
        value = int(row[column])
    except (KeyError, TypeError, ValueError):
        malformed.append(location)
        raise ValueError(location) from None
    return value


def _scan_rejected_numeric_cells(
    rows: Sequence[dict[str, str]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    text_columns = {"status", "error_type", "error_message"}
    optional_numeric_columns = {
        "min1_x_m",
        "min1_y_m",
        "min2_x_m",
        "min2_y_m",
        "min3_x_m",
        "min3_y_m",
        "min_pairwise_distance_m",
        "relative_free_residual",
    }
    numeric_columns = [item for item in REJECTED_CSV_COLUMNS if item not in text_columns]
    malformed = []
    nonfinite = []
    for row_index, row in enumerate(rows, start=2):
        for column in numeric_columns:
            text = row.get(column, "")
            if text == "" and column in optional_numeric_columns:
                continue
            location = f"rejected:{row_index}:{column}"
            try:
                value = float(text)
            except (TypeError, ValueError):
                malformed.append(location)
                continue
            if not np.isfinite(value):
                nonfinite.append(location)
    return tuple(malformed), tuple(nonfinite)


def _stack_or_empty(
    values: Sequence[NDArray[np.float64]],
    shape: tuple[int, int],
) -> NDArray[np.float64]:
    return np.stack(values) if values else np.empty((0, *shape), dtype=float)


def _duplicate_full_rows(rows: Sequence[dict[str, str]], schema: Sequence[str]) -> int:
    keys = [tuple(row.get(column, "") for column in schema) for row in rows]
    return len(keys) - len(set(keys))


def _duplicate_scalar_count(values: NDArray[np.int64]) -> int:
    return int(values.size - np.unique(values).size)


def _duplicate_array_rows(values: NDArray[np.float64]) -> int:
    if not values.size:
        return 0
    flattened = values.reshape(values.shape[0], -1)
    return int(flattened.shape[0] - np.unique(flattened, axis=0).shape[0])


def _domain_checks(
    fem_displacements_m: NDArray[np.float64],
    minima_positions_m: NDArray[np.float64],
) -> tuple[int, int, float, float]:
    geometry_config = real_scale_geometry_config()
    domain_violations = 0
    invalid_geometries = 0
    electrode_clearances = []
    outer_clearances = []
    for displacements, minima in zip(
        fem_displacements_m,
        minima_positions_m,
        strict=True,
    ):
        try:
            geometry = build_geometry_from_absolute_displacements(
                geometry_config,
                displacements,
            )
        except ValueError:
            invalid_geometries += 1
            continue
        domain_violations += int(np.count_nonzero(~geometry.contains_points(minima)))
        distances = np.linalg.norm(
            minima[:, np.newaxis, :] - geometry.centers_m[np.newaxis, :, :],
            axis=2,
        )
        electrode_clearances.extend(
            np.min(distances, axis=1) - geometry_config.electrode_radius_m
        )
        outer_clearances.extend(
            geometry_config.outer_radius_m - np.linalg.norm(minima, axis=1)
        )
    return (
        domain_violations,
        invalid_geometries,
        _finite_min(np.asarray(electrode_clearances)),
        _finite_min(np.asarray(outer_clearances)),
    )


def _build_statistics(
    raw: NDArray[np.float64],
    fem: NDArray[np.float64],
    minima: NDArray[np.float64],
    pairwise: NDArray[np.float64],
) -> tuple[SummaryStatistic, ...]:
    records = [
        _stat("wolfram_displacement", "all", raw.ravel()),
        _stat("wolfram_displacement", "x", raw[:, :, 0].ravel()),
        _stat("wolfram_displacement", "y", raw[:, :, 1].ravel()),
        _stat("fem_displacement", "all", fem.ravel()),
        _stat("fem_displacement", "x", fem[:, :, 0].ravel()),
        _stat("fem_displacement", "y", fem[:, :, 1].ravel()),
        _stat("minimum_coordinate", "all", minima.ravel()),
        _stat("minimum_coordinate", "x", minima[:, :, 0].ravel()),
        _stat("minimum_coordinate", "y", minima[:, :, 1].ravel()),
        _stat("minimum_radius", "all", np.linalg.norm(minima, axis=2).ravel()),
        _stat("min_pairwise_distance", "all", pairwise),
    ]
    for index in range(3):
        records.extend(
            (
                _stat(
                    f"minimum_{index + 1}_coordinate",
                    "x",
                    minima[:, index, 0],
                ),
                _stat(
                    f"minimum_{index + 1}_coordinate",
                    "y",
                    minima[:, index, 1],
                ),
                _stat(
                    f"minimum_{index + 1}_radius",
                    "r",
                    np.linalg.norm(minima[:, index, :], axis=1),
                ),
            )
        )
    return tuple(records)


def _stat(quantity: str, component: str, values: ArrayLike) -> SummaryStatistic:
    array = np.asarray(values, dtype=float)
    if not array.size or not np.all(np.isfinite(array)):
        raise ValueError(f"cannot summarize empty or nonfinite {quantity}/{component}")
    return SummaryStatistic(
        quantity=quantity,
        component=component,
        count=int(array.size),
        unit="m",
        minimum=float(np.min(array)),
        maximum=float(np.max(array)),
        mean=float(np.mean(array)),
        std=float(np.std(array)),
        median=float(np.median(array)),
    )


def _critical_issues(**values: object) -> tuple[str, ...]:
    issues = []
    if values["clean_schema"] != CLEAN_CSV_COLUMNS:
        issues.append("clean CSV schema does not match CLEAN_CSV_COLUMNS")
    if values["rejected_schema"] != REJECTED_CSV_COLUMNS:
        issues.append("rejected CSV schema does not match REJECTED_CSV_COLUMNS")
    clean_rows = int(values["clean_rows"])
    rejected_rows = int(values["rejected_rows"])
    if int(values["valid_rows"]) != clean_rows:
        issues.append("one or more clean rows could not be parsed completely")
    summary = values["summary"]
    if not isinstance(summary, dict):
        issues.append("summary JSON is not an object")
    else:
        expected = {
            "clean_samples": clean_rows,
            "rejected_samples": rejected_rows,
            "completed_samples": clean_rows + rejected_rows,
            "requested_samples": clean_rows + rejected_rows,
        }
        for key, expected_value in expected.items():
            if summary.get(key) != expected_value:
                issues.append(
                    f"summary {key}={summary.get(key)!r}, expected {expected_value}"
                )
        if summary.get("max_displacement_m") != QA_MAX_DISPLACEMENT_M:
            issues.append("summary max_displacement_m is not 0.0005 m")
        if (
            summary.get("ambiguous_minimum_distance_m")
            != DEFAULT_AMBIGUOUS_MINIMUM_DISTANCE_M
        ):
            issues.append("summary ambiguity threshold is not 0.00015 m")
        if summary.get("reference_row5_used") is not False:
            issues.append("summary does not confirm reference row 5 exclusion")
    count_messages = {
        "malformed": "malformed numeric cells",
        "nonfinite": "nonfinite numeric/JSON values",
        "duplicate_full": "duplicate complete clean rows",
        "duplicate_ids": "duplicate sample IDs",
        "duplicate_inputs": "duplicate Wolfram displacement vectors",
        "bound_violations": "displacement-bound violations",
        "transform_violations": "Wolfram-to-FEM transform violations",
        "search_violations": "minima outside the 8 mm search square",
        "domain_violations": "minima outside the displaced vacuum domain",
        "invalid_geometries": "invalid displaced geometries",
        "pairwise_violations": "clean rows below the pairwise-distance threshold",
        "pairwise_mismatches": "stored/recomputed pairwise-distance mismatches",
        "polar_violations": "polar-order violations",
    }
    for key, label in count_messages.items():
        item = values[key]
        count = len(item) if isinstance(item, tuple) else int(item)
        if count:
            issues.append(f"{count} {label}")
    statuses = values["statuses"]
    if any(item != "clean" for item in statuses):
        issues.append("clean CSV contains a non-clean status")
    sample_ids = np.asarray(values["sample_ids"])
    if sample_ids.size and not np.array_equal(
        np.sort(sample_ids),
        np.arange(1, clean_rows + 1),
    ):
        issues.append("sample IDs are not exactly 1..clean_row_count")
    return tuple(issues)


def _distribution_observations(
    summary: dict[str, object],
    raw: NDArray[np.float64],
    minima: NDArray[np.float64],
    pairwise: NDArray[np.float64],
    rejected_candidate_counts: NDArray[np.int64],
    sample_ids: NDArray[np.int64],
    rejected_rows: int,
) -> tuple[str, ...]:
    observations = []
    maximum = QA_MAX_DISPLACEMENT_M
    expected_std = maximum / np.sqrt(3.0)
    observed_std = float(np.std(raw))
    observations.append(
        "Aggregate Wolfram displacement standard deviation is "
        f"{1.0e6 * observed_std:.6g} µm versus {1.0e6 * expected_std:.6g} µm "
        "for an ideal continuous uniform distribution."
    )
    minima_mean = np.mean(minima, axis=(0, 1))
    observations.append(
        "Aggregate minimum-coordinate mean is "
        f"({1.0e3 * minima_mean[0]:.6g}, {1.0e3 * minima_mean[1]:.6g}) mm; "
        "small nonzero finite-sample offsets are not a schema defect."
    )
    observations.append(
        f"The closest minimum pair is {1.0e3 * np.min(pairwise):.6g} mm, "
        f"{np.min(pairwise) / DEFAULT_AMBIGUOUS_MINIMUM_DISTANCE_M:.3f} times "
        "the 0.15 mm rejection threshold."
    )
    closest_index = int(np.argmin(pairwise))
    observations.append(
        f"{np.count_nonzero(pairwise < 1.0e-3)} rows are below 1 mm and "
        f"{np.count_nonzero(pairwise < 2.0e-3)} are below 2 mm. The closest is "
        f"sample {sample_ids[closest_index]} with "
        f"{rejected_candidate_counts[closest_index]} additional robust-rejected "
        "candidates; it remains clean under the documented threshold."
    )
    nonzero = int(np.count_nonzero(rejected_candidate_counts))
    observations.append(
        f"{nonzero} clean rows contain at least one candidate rejected by robust "
        "quality rules; this is compatible with exactly three robust-accepted minima."
    )
    if rejected_rows == 0:
        observations.append(
            "No production row exercised the rejected split. The rejection path is "
            "unit-tested, but rare ambiguous cases remain possible in larger samples."
        )
    observations.append(
        "Polar-angle labels are deterministic but have the usual 0/2π seam; a model "
        "near that seam can see a label permutation despite continuous physics."
    )
    observations.append(
        f"N={raw.shape[0]} is suitable for inverse-model experiments, not a final "
        "coverage claim for the full eight-dimensional displacement space."
    )
    return tuple(observations)


def _write_statistics_csv(
    path: Path,
    statistics: Sequence[SummaryStatistic],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(SUMMARY_STAT_COLUMNS))
        writer.writeheader()
        for item in statistics:
            writer.writerow(
                {
                    "quantity": item.quantity,
                    "component": item.component,
                    "count": item.count,
                    "unit": item.unit,
                    "minimum": item.minimum,
                    "maximum": item.maximum,
                    "mean": item.mean,
                    "std": item.std,
                    "median": item.median,
                }
            )


def _write_displacement_histogram(audit: DatasetQAAudit, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    values_um = 1.0e6 * audit.raw_wolfram_displacements_m.ravel()
    axis.hist(values_um, bins=30, color="#2563EB", alpha=0.82, edgecolor="white")
    axis.axvline(-500.0, color="#991B1B", linestyle="--", linewidth=1.2)
    axis.axvline(500.0, color="#991B1B", linestyle="--", linewidth=1.2)
    axis.set(title="Wolfram-order displacement coordinates", xlabel="Displacement (µm)", ylabel="Count")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_minima_scatter(audit: DatasetQAAudit, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.3, 6.0))
    colors = ("#2563EB", "#D97706", "#059669")
    for index, color in enumerate(colors):
        points = 1.0e3 * audit.minima_positions_m[:, index, :]
        axis.scatter(
            points[:, 0],
            points[:, 1],
            s=9,
            alpha=0.48,
            color=color,
            label=f"min{index + 1}",
        )
    extent_mm = 1.0e3 * REAL_SEARCH_HALF_WIDTH_M
    axis.set_xlim(-extent_mm, extent_mm)
    axis.set_ylim(-extent_mm, extent_mm)
    axis.set_aspect("equal", adjustable="box")
    axis.set(title="Angle-sorted pseudopotential minima", xlabel="x (mm)", ylabel="y (mm)")
    axis.legend(loc="upper right")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_minima_histogram(audit: DatasetQAAudit, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), sharey=True)
    flattened = 1.0e3 * audit.minima_positions_m
    axes[0].hist(flattened[:, :, 0].ravel(), bins=32, color="#7C3AED", alpha=0.82)
    axes[1].hist(flattened[:, :, 1].ravel(), bins=32, color="#0891B2", alpha=0.82)
    axes[0].set(title="Minimum x coordinates", xlabel="x (mm)", ylabel="Count")
    axes[1].set(title="Minimum y coordinates", xlabel="y (mm)")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_pairwise_histogram(audit: DatasetQAAudit, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    values_mm = 1.0e3 * audit.reported_pairwise_distances_m
    axis.hist(values_mm, bins=30, color="#0F766E", alpha=0.84, edgecolor="white")
    axis.axvline(0.15, color="#B91C1C", linestyle="--", linewidth=1.4, label="0.15 mm rejection threshold")
    axis.set(title="Minimum pairwise separation", xlabel="Distance (mm)", ylabel="Count")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _markdown_report(audit: DatasetQAAudit) -> str:
    stat = {(item.quantity, item.component): item for item in audit.statistics}
    raw = stat[("wolfram_displacement", "all")]
    minima_x = stat[("minimum_coordinate", "x")]
    minima_y = stat[("minimum_coordinate", "y")]
    pairwise = stat[("min_pairwise_distance", "all")]
    checks = (
        ("Clean schema exact", audit.clean_schema == CLEAN_CSV_COLUMNS),
        ("Rejected schema exact", audit.rejected_schema == REJECTED_CSV_COLUMNS),
        ("All clean rows parsed", audit.valid_clean_row_count == audit.clean_row_count),
        ("No malformed/nonfinite values", not audit.malformed_numeric_cells and not audit.nonfinite_numeric_cells),
        ("No duplicate full rows/IDs/inputs", audit.duplicate_full_rows == audit.duplicate_sample_ids == audit.duplicate_wolfram_inputs == 0),
        ("All displacement coordinates within bounds", audit.displacement_bound_violations == 0),
        ("Wolfram transform exact", audit.transform_violations == 0),
        ("All minima inside ±8 mm search square", audit.search_region_violations == 0),
        ("All minima inside displaced vacuum domain", audit.vacuum_domain_violations == 0 and audit.invalid_geometries == 0),
        ("All clean separations ≥0.15 mm", audit.pairwise_threshold_violations == 0),
        ("Stored pairwise values recompute exactly", audit.pairwise_value_mismatches == 0),
        ("Polar-order violations are zero", audit.polar_order_violations == 0),
    )
    lines = [
        f"# Generated N={audit.clean_row_count + audit.rejected_row_count} dataset QA",
        "",
        f"**Conclusion: {'ML-ready for a first inverse-model experiment' if audit.ml_ready else 'NOT ML-ready'}.**",
        "",
        "This audit is read-only with respect to the generated dataset. It does not run FEM, generate samples, calibrate physics, or train a model.",
        "",
        "## Integrity checks",
        "",
        f"Clean rows: **{audit.clean_row_count}**; rejected rows: **{audit.rejected_row_count}**; completely parsed clean rows: **{audit.valid_clean_row_count}**.",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    lines.extend(f"| {label} | {'PASS' if passed else 'FAIL'} |" for label, passed in checks)
    lines.extend(
        (
            "",
            f"Exact duplicate minima-output triples: **{audit.duplicate_minima_outputs}**.",
            f"Minimum electrode-surface clearance: **{1.0e3 * audit.minimum_electrode_clearance_m:.6g} mm**; minimum outer-boundary clearance: **{1.0e3 * audit.minimum_outer_clearance_m:.6g} mm**.",
            f"Largest absolute minimum coordinate: **{1.0e3 * audit.maximum_absolute_minimum_coordinate_m:.6g} mm** versus the 8 mm search half-width.",
            f"Maximum stored/recomputed pairwise-distance discrepancy: **{audit.maximum_pairwise_value_error_m:.3g} m**.",
            "",
            "## Requested distribution statistics",
            "",
            "| Quantity | Minimum | Maximum | Mean | Standard deviation | Median |",
            "|---|---:|---:|---:|---:|---:|",
            f"| Wolfram displacement coordinates (µm) | {1.0e6 * raw.minimum:.9g} | {1.0e6 * raw.maximum:.9g} | {1.0e6 * raw.mean:.9g} | {1.0e6 * raw.std:.9g} | {1.0e6 * raw.median:.9g} |",
            f"| Minimum x coordinates (mm) | {1.0e3 * minima_x.minimum:.9g} | {1.0e3 * minima_x.maximum:.9g} | {1.0e3 * minima_x.mean:.9g} | {1.0e3 * minima_x.std:.9g} | {1.0e3 * minima_x.median:.9g} |",
            f"| Minimum y coordinates (mm) | {1.0e3 * minima_y.minimum:.9g} | {1.0e3 * minima_y.maximum:.9g} | {1.0e3 * minima_y.mean:.9g} | {1.0e3 * minima_y.std:.9g} | {1.0e3 * minima_y.median:.9g} |",
            f"| Minimum pairwise distance (mm) | {1.0e3 * pairwise.minimum:.9g} | {1.0e3 * pairwise.maximum:.9g} | {1.0e3 * pairwise.mean:.9g} | {1.0e3 * pairwise.std:.9g} | {1.0e3 * pairwise.median:.9g} |",
            "",
            "The full SI-unit table, including x/y and per-label summaries, is in `summary_stats.csv`.",
            "",
            "## Duplicate and value diagnostics",
            "",
            f"- Duplicate complete rows: {audit.duplicate_full_rows}.",
            f"- Duplicate sample IDs: {audit.duplicate_sample_ids}.",
            f"- Duplicate Wolfram displacement vectors: {audit.duplicate_wolfram_inputs}.",
            f"- Duplicate minima-output triples: {audit.duplicate_minima_outputs}.",
            f"- Malformed numeric cells: {len(audit.malformed_numeric_cells)}.",
            f"- NaN/inf cells or JSON values: {len(audit.nonfinite_numeric_cells)}.",
            "",
            "## Distribution observations and cautions",
            "",
        )
    )
    lines.extend(f"- {item}" for item in audit.observations)
    lines.extend(("", "## Plots", ""))
    lines.extend(
        (
            "- `plots/displacement_coordinates.png`",
            "- `plots/minima_positions.png`",
            "- `plots/minima_coordinates.png`",
            "- `plots/minimum_pairwise_distance.png`",
        )
    )
    lines.extend(("", "## ML readiness", ""))
    if audit.critical_issues:
        lines.append("Training is blocked by:")
        lines.extend(f"- {item}" for item in audit.critical_issues)
    else:
        lines.extend(
            (
                "The dataset passes every file-integrity, numerical, geometry, separation, and deterministic-label check and is safe for a **first inverse-model experiment**.",
                f"This is not evidence that N={audit.clean_row_count} fully covers the eight-dimensional input space or that the four-electrode FEM is a complete physical model of every reference branch. Keep row-5-like ambiguities quarantined if they appear in future generation.",
            )
        )
    return "\n".join(lines) + "\n"


def _finite_min(values: NDArray[np.float64]) -> float:
    return float(np.min(values)) if values.size else float("nan")


def _finite_max(values: NDArray[np.float64]) -> float:
    return float(np.max(values)) if values.size else float("nan")


def build_parser() -> argparse.ArgumentParser:
    """Build the focused generated-dataset QA command line."""

    parser = argparse.ArgumentParser(
        prog="rf-trap-audit-dataset",
        description="Audit an existing generated clean/rejected dataset split.",
    )
    parser.add_argument(
        "--clean-csv",
        type=Path,
        default=Path("validation_results/generated_dataset/synthetic_clean.csv"),
    )
    parser.add_argument(
        "--rejected-csv",
        type=Path,
        default=Path("validation_results/generated_dataset/synthetic_rejected.csv"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("validation_results/generated_dataset/synthetic_summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation_results/generated_dataset_qa"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the read-only dataset audit and write its requested artifacts."""

    arguments = build_parser().parse_args(argv)
    audit = audit_generated_dataset(
        arguments.clean_csv,
        arguments.rejected_csv,
        arguments.summary_json,
    )
    paths = write_dataset_qa_outputs(audit, arguments.output_dir)
    print(f"ml_ready={audit.ml_ready}")
    print(f"clean_rows={audit.clean_row_count}")
    print(f"rejected_rows={audit.rejected_row_count}")
    print(f"critical_issues={len(audit.critical_issues)}")
    print(f"polar_order_violations={audit.polar_order_violations}")
    print(f"report={paths.report_markdown}")
    print(f"summary_stats={paths.summary_stats_csv}")
    return 0 if audit.ml_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
