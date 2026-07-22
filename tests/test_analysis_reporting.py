"""Lightweight tests for saved-result analysis and the ML-only learning curve."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from rf_trap_forward.error_analysis import (
    build_closed_loop_analysis,
    load_prediction_error_data,
    write_error_analysis_outputs,
)
from rf_trap_forward.inverse_training import InverseDataset, TARGET_COLUMNS
from rf_trap_forward.learning_curve import (
    effective_training_rows,
    run_learning_curve_experiment,
    validate_learning_curve_sizes,
    write_learning_curve_outputs,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write one tiny test CSV with stable columns."""

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _prediction_row(model: str, sample_id: int, error_um: float) -> dict[str, object]:
    """Create one complete prediction row with a uniform signed error."""

    row: dict[str, object] = {"model": model, "sample_id": sample_id}
    for column in TARGET_COLUMNS:
        row[f"true_{column}"] = sample_id * 1.0e-6
        row[f"predicted_{column}"] = (sample_id + error_um) * 1.0e-6
    return row


def _dataset_row(sample_id: int) -> dict[str, object]:
    """Create the context needed to join one closed-loop case."""

    row: dict[str, object] = {"sample_id": sample_id}
    for index, column in enumerate(TARGET_COLUMNS):
        row[column] = (sample_id + index) * 1.0e-6
    minima = (0.001, 0.002, -0.002, 0.001, 0.0005, -0.0025)
    for column, value in zip(
        (
            "min1_x_m",
            "min1_y_m",
            "min2_x_m",
            "min2_y_m",
            "min3_x_m",
            "min3_y_m",
        ),
        minima,
        strict=True,
    ):
        row[column] = value
    row["min_pairwise_distance_m"] = 0.003
    return row


def _closed_loop_row(sample_id: int, errors: tuple[float, float, float]) -> dict[str, object]:
    """Create one included closed-loop result row."""

    row: dict[str, object] = {
        "sample_id": sample_id,
        "status": "ok",
        "included_in_error_summary": "True",
        "exactly_three_robust_minima": "True",
        "match1_error_um": errors[0],
        "match2_error_um": errors[1],
        "match3_error_um": errors[2],
        "row_mean_error_um": np.mean(errors),
        "row_median_error_um": np.median(errors),
        "row_max_error_um": np.max(errors),
        "min_pairwise_distance_m": 0.003 + sample_id * 1.0e-5,
    }
    for index, column in enumerate(TARGET_COLUMNS):
        row[f"predicted_{column}"] = (sample_id + index + 0.25) * 1.0e-6
    minima = (0.001, 0.002, -0.002, 0.001, 0.0005, -0.0025 - sample_id * 1.0e-6)
    for column, value in zip(
        (
            "min1_x_m",
            "min1_y_m",
            "min2_x_m",
            "min2_y_m",
            "min3_x_m",
            "min3_y_m",
        ),
        minima,
        strict=True,
    ):
        row[f"true_{column}"] = value
    return row


def test_saved_error_analysis_recomputes_metrics_and_writes_outputs(
    tmp_path: Path,
) -> None:
    """The analysis must filter MLP rows, join IDs, and preserve exact errors."""

    predictions = tmp_path / "predictions.csv"
    dataset = tmp_path / "dataset.csv"
    closed = tmp_path / "closed.csv"
    _write_csv(
        predictions,
        [
            _prediction_row("ridge", 1, 50.0),
            _prediction_row("mlp", 1, 10.0),
            _prediction_row("mlp", 2, -20.0),
        ],
    )
    _write_csv(dataset, [_dataset_row(1), _dataset_row(2)])
    _write_csv(
        closed,
        [_closed_loop_row(1, (10.0, 20.0, 30.0)), _closed_loop_row(2, (40.0, 50.0, 60.0))],
    )
    prediction_data = load_prediction_error_data(predictions)
    assert prediction_data.sample_ids.tolist() == [1, 2]
    assert np.allclose(prediction_data.errors_um[0], 10.0)
    assert np.allclose(prediction_data.errors_um[1], -20.0)
    closed_data = build_closed_loop_analysis(closed, dataset)
    assert closed_data.matched_errors_um.tolist() == [10, 20, 30, 40, 50, 60]
    paths = write_error_analysis_outputs(
        prediction_data,
        closed_data,
        tmp_path / "analysis",
        prediction_source=predictions,
        closed_loop_source=closed,
        dataset_source=dataset,
    )
    assert paths.worst_cases_csv.is_file()
    assert paths.summary_json.is_file()
    assert len(list(paths.plot_directory.glob("*.png"))) == 7


def test_learning_curve_sizes_and_smoke_fit_are_deterministic(tmp_path: Path) -> None:
    """Nested subset counts and a tiny MLP curve must use one fixed test set."""

    assert effective_training_rows(1000) == 800
    assert effective_training_rows(29995) == 23996
    assert validate_learning_curve_sizes((20, 60), 60) == (20, 60)
    rng = np.random.default_rng(7)
    X = rng.normal(scale=1.0e-3, size=(60, 6))
    weights = rng.normal(size=(6, 8))
    y = X @ weights * 0.05
    dataset = InverseDataset(np.arange(1, 61), X, y)
    result = run_learning_curve_experiment(
        dataset,
        dataset_sizes=(20, 60),
        smoke_test=True,
        random_state=42,
    )
    assert [point.train_rows for point in result.points] == [16, 48]
    assert [point.fixed_test_rows for point in result.points] == [12, 12]
    paths = write_learning_curve_outputs(
        result, tmp_path / "curve", dataset_source="synthetic_test.csv"
    )
    assert paths.metrics_csv.is_file()
    assert paths.summary_json.is_file()
    assert not list((tmp_path / "curve").glob("*.joblib"))
