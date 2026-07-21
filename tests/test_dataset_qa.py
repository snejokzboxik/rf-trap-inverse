"""Lightweight tests for generated-dataset integrity auditing."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from rf_trap_forward.dataset_qa import (
    SUMMARY_STAT_COLUMNS,
    audit_generated_dataset,
    count_polar_order_violations,
    write_dataset_qa_outputs,
)
from rf_trap_forward.synthetic_dataset import (
    SyntheticDatasetConfig,
    SyntheticSolveResult,
    generate_synthetic_dataset,
    write_synthetic_dataset,
)


def _clean_worker(*_: object) -> SyntheticSolveResult:
    return SyntheticSolveResult(
        minima_positions_m=np.asarray(
            ((1.0e-3, 2.0e-3), (-2.0e-3, 0.0), (1.0e-3, -2.0e-3))
        ),
        accepted_candidate_count=3,
        rejected_candidate_count=1,
        total_candidate_count=4,
        selected_interpolation_sensitive_count=0,
        node_count=100,
        triangle_count=180,
        relative_free_residual=1.0e-15,
        runtime_seconds=0.1,
    )


def _small_dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    result = generate_synthetic_dataset(
        SyntheticDatasetConfig(n=2, seed=123, batch_size=1),
        worker=_clean_worker,
    )
    paths = write_synthetic_dataset(result, tmp_path)
    return paths.clean_csv, paths.rejected_csv, paths.summary_json


def test_valid_generated_split_passes_and_writes_requested_outputs(
    tmp_path: Path,
) -> None:
    """A small valid split must pass every blocking QA gate and write plots."""

    clean, rejected, summary = _small_dataset(tmp_path / "dataset")
    audit = audit_generated_dataset(clean, rejected, summary)
    assert audit.ml_ready
    assert audit.clean_row_count == 2
    assert audit.rejected_row_count == 0
    assert audit.polar_order_violations == 0
    paths = write_dataset_qa_outputs(audit, tmp_path / "qa")
    assert paths.report_markdown.is_file()
    assert paths.summary_stats_csv.is_file()
    assert all(
        item.is_file()
        for item in (
            paths.displacement_histogram_png,
            paths.minima_scatter_png,
            paths.minima_histogram_png,
            paths.pairwise_histogram_png,
        )
    )
    with paths.summary_stats_csv.open(encoding="utf-8", newline="") as stream:
        assert tuple(csv.DictReader(stream).fieldnames or ()) == SUMMARY_STAT_COLUMNS


def test_polar_order_violation_counter_detects_permuted_labels() -> None:
    """The label audit must fail when angles decrease within one sample."""

    ordered = np.asarray(
        (((1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0)),)
    )
    permuted = ordered[:, (1, 0, 2), :]
    assert count_polar_order_violations(ordered) == 0
    assert count_polar_order_violations(permuted) == 1


def test_audit_detects_nonfinite_and_out_of_range_input(tmp_path: Path) -> None:
    """NaN and a 600 µm displacement must block ML readiness."""

    clean, rejected, summary = _small_dataset(tmp_path / "dataset")
    with clean.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    rows[0]["w1_dx_m"] = "0.0006"
    rows[1]["min1_x_m"] = "nan"
    with clean.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    audit = audit_generated_dataset(clean, rejected, summary)
    assert not audit.ml_ready
    assert audit.nonfinite_numeric_cells
    assert audit.valid_clean_row_count == 1
    assert audit.displacement_bound_violations == 1
