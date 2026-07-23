"""Lightweight tests for export-row FEM diagnostics."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from rf_trap_forward.dataset import sort_points_by_polar_angle
from rf_trap_forward.verify_export_row_fem import (
    ExportRow,
    load_export_rows,
    select_export_rows,
    wrong_direct_fem_mapping,
)
from rf_trap_forward.absolute_validation import wolfram_to_fem_absolute_displacements_m


def test_canonical_wolfram_transform_matches_documented_mapping() -> None:
    wolfram = np.asarray(((1, 2), (3, 4), (5, 6), (7, 8)), dtype=float)
    expected = np.asarray(((-5, -6), (-1, -2), (-7, -8), (-3, -4)))
    assert np.array_equal(
        wolfram_to_fem_absolute_displacements_m(wolfram), expected
    )
    assert np.array_equal(wrong_direct_fem_mapping(wolfram), wolfram)


def test_export_row_minima_selection_preserves_csv_order(tmp_path: Path) -> None:
    path = tmp_path / "export.csv"
    w_columns = [
        f"true_w{electrode}_{component}_m"
        for electrode in range(1, 5)
        for component in ("dx", "dy")
    ]
    minima_columns = [
        f"min{minimum}_{component}_m"
        for minimum in range(1, 4)
        for component in ("x", "y")
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", *w_columns, *minima_columns])
        writer.writeheader()
        for sample_id in (10, 20):
            writer.writerow(
                {
                    "sample_id": sample_id,
                    **dict(zip(w_columns, np.arange(8) * 1.0e-6, strict=True)),
                    **dict(zip(minima_columns, np.arange(6) * 1.0e-3, strict=True)),
                }
            )
    rows = load_export_rows(path)
    assert [row.sample_id for row in rows] == [10, 20]
    assert select_export_rows(rows, row_indices=[1])[0].sample_id == 20
    assert select_export_rows(rows, sample_ids=[10])[0].row_index == 0


def test_existing_polar_sort_helper_is_the_expected_minima_convention() -> None:
    points = np.asarray(((1.0, 0.0), (-1.0, 0.0), (0.0, -1.0)))
    sorted_points = sort_points_by_polar_angle(points)
    assert np.all(np.diff(np.mod(np.arctan2(sorted_points[:, 1], sorted_points[:, 0]), 2 * np.pi)) >= 0)
