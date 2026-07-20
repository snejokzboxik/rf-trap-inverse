"""Tests for Mathematica reference-dataset ingestion and export."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from rf_trap_forward.dataset import (
    ReferenceDataset,
    displacements_relative_to_electrode1,
    export_reference_dataset,
    load_reference_dataset,
    parse_reference_row,
    sort_points_by_polar_angle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_parse_one_complete_reference_row() -> None:
    """One full mapping must become four displacement and three minima pairs."""

    line = (
        "{{-0.0002, -0.0003}, {-0.0001, -0.0004}, "
        "{0.0003, 0.0002}, {0.0004, 0.00001}} -> "
        "{{0.0038, 0.0008}, {-0.0045, -0.0003}, {0.0004, -0.0002}}"
    )
    displacements, minima = parse_reference_row(line)
    assert displacements.shape == (4, 2)
    assert minima.shape == (3, 2)
    np.testing.assert_allclose(displacements[0], [-0.0002, -0.0003])
    np.testing.assert_allclose(minima[-1], [0.0004, -0.0002])


def test_parse_mathematica_scientific_notation() -> None:
    """Mathematica ``*^`` exponents must parse without changing their value."""

    line = (
        "{{5.449003089273541*^-6, 2*^+3}, {0, 0}, {0, 0}, {0, 0}} -> "
        "{{1*^-3, 0}, {0, 1*^-3}, {-1*^-3, 0}}"
    )
    displacements, minima = parse_reference_row(line)
    assert displacements[0, 0] == 5.449003089273541e-6
    assert displacements[0, 1] == 2.0e3
    assert minima[0, 0] == 1.0e-3


def test_reference_file_row_count_and_shapes() -> None:
    """The supplied file must retain its verified 326-row structure."""

    dataset = load_reference_dataset(PROJECT_ROOT / "Data.txt")
    assert dataset.row_count == 326
    assert dataset.raw_displacements_m.shape == (326, 4, 2)
    assert dataset.raw_minima_absolute_m.shape == (326, 3, 2)
    assert len(dataset.mathematica_notation_rows) == 57


def test_convert_raw_8d_displacements_to_electrode1_relative_6d() -> None:
    """Electrodes 2--4 must be translated by electrode 1 and flatten to six values."""

    raw = np.asarray(
        [
            [10.0, -20.0],
            [13.0, -18.0],
            [4.0, -30.0],
            [12.0, -15.0],
        ]
    )
    relative = displacements_relative_to_electrode1(raw)
    np.testing.assert_array_equal(relative, [[3.0, 2.0], [-6.0, -10.0], [2.0, 5.0]])


def test_minima_are_sorted_by_forward_polar_angle_convention() -> None:
    """Angular ordering must use ``atan2`` mapped to the half-open ``[0, 2π)`` range."""

    points = np.asarray([[0.0, -1.0], [-1.0, 0.0], [1.0, 0.0]])
    sorted_points = sort_points_by_polar_angle(points)
    np.testing.assert_array_equal(
        sorted_points,
        [[1.0, 0.0], [-1.0, 0.0], [0.0, -1.0]],
    )


def test_csv_and_npz_export_keep_raw_and_relative_frames(tmp_path: Path) -> None:
    """Both exports must retain raw arrays, derived 6D inputs, and both minima frames."""

    raw_displacements = np.asarray(
        [
            [[1.0, 2.0], [3.0, 5.0], [7.0, 11.0], [13.0, 17.0]],
            [[-1.0, -2.0], [2.0, 4.0], [6.0, 8.0], [10.0, 12.0]],
        ]
    ) * 1.0e-6
    raw_minima = np.asarray(
        [
            [[0.0, -1.0], [-1.0, 0.0], [1.0, 0.0]],
            [[1.0, 1.0], [-1.0, 1.0], [0.0, -1.0]],
        ]
    ) * 1.0e-3
    dataset = ReferenceDataset(raw_displacements, raw_minima)
    paths = export_reference_dataset(
        dataset,
        tmp_path / "reference.csv",
        tmp_path / "reference.npz",
        primary_minima_frame="electrode1-relative",
    )

    with paths.csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert rows[0]["primary_minima_frame"] == "electrode1-relative"
    assert "raw_d1_x_m" in rows[0]
    assert "relative_d4_y_m" in rows[0]
    assert "minimum_absolute_sorted_3_y_m" in rows[0]
    assert "minimum_relative_sorted_3_y_m" in rows[0]

    with np.load(paths.npz_path) as archive:
        assert archive["raw_displacements_m"].shape == (2, 4, 2)
        assert archive["relative_displacements_flat_m"].shape == (2, 6)
        assert archive["raw_minima_absolute_m"].shape == (2, 3, 2)
        assert archive["minima_relative_angle_sorted_m"].shape == (2, 3, 2)
        assert archive["primary_minima_frame"].item() == "electrode1-relative"
