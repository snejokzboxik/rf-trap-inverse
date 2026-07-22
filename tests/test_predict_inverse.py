"""Lightweight tests for the direct inverse-model prediction interface."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from app_inverse_model_tk import copy_text_to_clipboard
from rf_trap_forward.export_prediction_dataset import load_prediction_model
from rf_trap_forward.dataset import sort_points_by_polar_angle
from rf_trap_forward.predict_inverse import (
    DEFAULT_MODEL_PATH,
    FEM_METRE_COLUMNS,
    FEM_MICROMETRE_COLUMNS,
    PREDICTED_WOLFRAM_METRE_COLUMNS,
    PREDICTED_WOLFRAM_MICROMETRE_COLUMNS,
    PREDICTION_OUTPUT_COLUMNS,
    canonical_order_minima,
    convert_minima_to_metres,
    load_minima_csv,
    normalize_minima_cli_args,
    parse_minima_string,
    predict_inverse,
    wolfram_to_fem_displacements_m,
    write_prediction_csv,
)
from rf_trap_forward.inverse_training import INPUT_COLUMNS


class _ZeroModel:
    def predict(self, minima_m: np.ndarray) -> np.ndarray:
        return np.zeros((minima_m.shape[0], 8), dtype=float)


class _InputEchoModel:
    def predict(self, minima_m: np.ndarray) -> np.ndarray:
        base = np.asarray(minima_m, dtype=float)[:, :1]
        return base + np.arange(8, dtype=float)


def test_minima_string_parser_and_millimetre_conversion() -> None:
    parsed = parse_minima_string("1,2;3,4;5,6", units="mm")
    assert parsed.shape == (3, 2)
    assert np.allclose(parsed, 1.0e-3 * np.asarray(((1, 2), (3, 4), (5, 6))))
    assert np.allclose(convert_minima_to_metres([1.0, -2.0], "mm"), [0.001, -0.002])


def test_negative_leading_minima_cli_value_is_bound_to_option() -> None:
    arguments = normalize_minima_cli_args(
        ["--minima", "-1,2;3,4;5,6", "--units", "mm"]
    )
    assert arguments == ["--minima=-1,2;3,4;5,6", "--units", "mm"]


def test_wolfram_to_fem_transform_is_negative_3142_order() -> None:
    wolfram = np.asarray(((1, 2), (3, 4), (5, 6), (7, 8)), dtype=float)
    expected = np.asarray(((-5, -6), (-1, -2), (-7, -8), (-3, -4)))
    assert np.array_equal(wolfram_to_fem_displacements_m(wolfram), expected)
    assert np.array_equal(
        wolfram_to_fem_displacements_m(wolfram.reshape(8)), expected.reshape(8)
    )


def test_canonical_minima_order_matches_existing_dataset_helper() -> None:
    points = np.asarray(((1.0, 0.0), (-1.0, 0.0), (0.0, -1.0)), dtype=float)
    assert np.array_equal(canonical_order_minima(points), sort_points_by_polar_angle(points))


def test_auto_sort_makes_permuted_inputs_equivalent_and_no_sort_does_not() -> None:
    points = np.asarray(((1.0, 0.0), (-1.0, 0.0), (0.0, -1.0)), dtype=float)
    permuted = points[[1, 2, 0]]
    model = _InputEchoModel()
    sorted_a = predict_inverse(model, points, sort_minima=True)
    sorted_b = predict_inverse(model, permuted, sort_minima=True)
    assert np.array_equal(sorted_a.minima_m, sorted_b.minima_m)
    assert np.array_equal(
        sorted_a.wolfram_displacements_m, sorted_b.wolfram_displacements_m
    )
    unsorted_a = predict_inverse(model, points, sort_minima=False)
    unsorted_b = predict_inverse(model, permuted, sort_minima=False)
    assert not np.array_equal(
        unsorted_a.wolfram_displacements_m, unsorted_b.wolfram_displacements_m
    )


def test_csv_input_is_sorted_before_model_prediction(tmp_path: Path) -> None:
    points = np.asarray(((1.0, 0.0), (-1.0, 0.0), (0.0, -1.0)), dtype=float)
    row = points[[1, 2, 0]].reshape(-1)
    path = tmp_path / "minima.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(INPUT_COLUMNS)
        writer.writerow(row)
    parsed = load_minima_csv(path)
    prediction = predict_inverse(_InputEchoModel(), parsed, sort_minima=True)
    assert np.array_equal(prediction.minima_m[0].reshape(3, 2), canonical_order_minima(points))


def test_copy_output_helper_uses_clipboard_callbacks() -> None:
    copied: list[str] = []
    copy_text_to_clipboard(lambda: copied.clear(), copied.append, "full output")
    assert copied == ["full output"]


def test_output_column_names_and_prediction_csv(tmp_path: Path) -> None:
    prediction = predict_inverse(_ZeroModel(), np.arange(6, dtype=float) * 1.0e-3)
    path = write_prediction_csv(prediction, tmp_path / "prediction.csv")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        assert tuple(reader.fieldnames or ()) == PREDICTION_OUTPUT_COLUMNS
    assert len(rows) == 1
    assert tuple(PREDICTION_OUTPUT_COLUMNS[:6]) == INPUT_COLUMNS
    assert len(PREDICTED_WOLFRAM_METRE_COLUMNS) == 8
    assert len(PREDICTED_WOLFRAM_MICROMETRE_COLUMNS) == 8
    assert len(FEM_METRE_COLUMNS) == 8
    assert len(FEM_MICROMETRE_COLUMNS) == 8


def test_saved_default_model_prediction_shape_if_available() -> None:
    if not DEFAULT_MODEL_PATH.is_file():
        pytest.skip(f"saved model is not available at {DEFAULT_MODEL_PATH}")
    model = load_prediction_model(DEFAULT_MODEL_PATH)
    minima_m = parse_minima_string(
        "-0.001596,0.003869;-0.001836,-0.003034;0.004218,-0.001076"
    )
    prediction = predict_inverse(model, minima_m)
    assert prediction.wolfram_displacements_m.shape == (1, 8)
    assert prediction.fem_displacements_m.shape == (1, 8)
    assert np.all(np.isfinite(prediction.wolfram_displacements_m))
