"""Tests for the first inverse-model baseline experiment."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from rf_trap_forward.inverse_training import (
    INPUT_COLUMNS,
    MODEL_NAMES,
    TARGET_COLUMNS,
    compute_inverse_metrics,
    load_inverse_dataset,
    split_inverse_dataset,
    train_inverse_baselines,
    write_inverse_training_outputs,
)


def _write_training_csv(path: Path, *, rows: int = 60) -> Path:
    generator = np.random.default_rng(9)
    X_m = generator.normal(scale=2.0e-3, size=(rows, 6))
    coefficients = generator.normal(scale=0.08, size=(6, 8))
    y_m = X_m @ coefficients
    fieldnames = (
        "sample_id",
        *INPUT_COLUMNS,
        *TARGET_COLUMNS,
        "f1_dx_m",
        "status",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames))
        writer.writeheader()
        for index in range(rows):
            record: dict[str, object] = {
                "sample_id": index + 1,
                "f1_dx_m": 99.0,
                "status": "clean",
            }
            record.update(
                {column: X_m[index, column_index] for column_index, column in enumerate(INPUT_COLUMNS)}
            )
            record.update(
                {column: y_m[index, column_index] for column_index, column in enumerate(TARGET_COLUMNS)}
            )
            writer.writerow(record)
    return path


def test_dataset_loading_uses_six_inputs_and_eight_wolfram_targets(
    tmp_path: Path,
) -> None:
    """The loader must use minima as X and raw W columns, never FEM columns, as y."""

    dataset = load_inverse_dataset(_write_training_csv(tmp_path / "clean.csv"))
    assert dataset.X_m.shape == (60, 6)
    assert dataset.y_m.shape == (60, 8)
    assert np.all(dataset.y_m != 99.0)
    assert np.array_equal(dataset.sample_ids, np.arange(1, 61))


def test_train_test_split_is_deterministic(tmp_path: Path) -> None:
    """Repeated 80/20 splits with state 42 must contain identical sample IDs."""

    dataset = load_inverse_dataset(_write_training_csv(tmp_path / "clean.csv"))
    first = split_inverse_dataset(dataset, test_size=0.2, random_state=42)
    second = split_inverse_dataset(dataset, test_size=0.2, random_state=42)
    assert np.array_equal(first.train_sample_ids, second.train_sample_ids)
    assert np.array_equal(first.test_sample_ids, second.test_sample_ids)
    assert first.X_train_m.shape == (48, 6)
    assert first.y_test_m.shape == (12, 8)


def test_metric_computation_converts_metres_to_micrometres() -> None:
    """A one-micrometre coordinate error must be reported as exactly one."""

    truth = np.zeros((3, 8), dtype=float)
    prediction = np.full((3, 8), 1.0e-6, dtype=float)
    metrics = compute_inverse_metrics(truth, prediction)
    assert np.isclose(metrics.overall_mae_um, 1.0)
    assert np.isclose(metrics.overall_rmse_um, 1.0)
    assert np.isclose(metrics.max_absolute_error_um, 1.0)
    assert np.allclose(metrics.per_output_mae_um, 1.0)
    assert np.allclose(metrics.per_electrode_vector_error_mean_um, np.sqrt(2.0))
    assert np.allclose(metrics.per_electrode_vector_error_max_um, np.sqrt(2.0))


def test_tiny_three_model_training_smoke_run(tmp_path: Path) -> None:
    """All requested models must fit and predict the eight targets on a small table."""

    dataset = load_inverse_dataset(_write_training_csv(tmp_path / "clean.csv", rows=48))
    result = train_inverse_baselines(
        dataset,
        test_size=0.2,
        random_state=42,
        smoke_test=True,
    )
    assert tuple(item.name for item in result.evaluations) == MODEL_NAMES
    assert result.split.X_test_m.shape == (10, 6)
    assert result.split.y_test_m.shape == (10, 8)
    for evaluation in result.evaluations:
        assert evaluation.predictions_m.shape == (10, 8)
        assert np.all(np.isfinite(evaluation.predictions_m))
        assert np.isfinite(evaluation.metrics.overall_mae_um)


def test_output_writer_can_skip_large_random_forest_artifact(tmp_path: Path) -> None:
    """All models remain evaluated when only random-forest serialization is skipped."""

    dataset = load_inverse_dataset(_write_training_csv(tmp_path / "clean.csv", rows=48))
    result = train_inverse_baselines(
        dataset, test_size=0.2, random_state=42, smoke_test=True
    )
    paths = write_inverse_training_outputs(
        result,
        tmp_path / "outputs",
        skip_model_artifacts=("random_forest",),
    )

    assert {path.name for path in paths.model_paths} == {"ridge.joblib", "mlp.joblib"}
    assert not (tmp_path / "outputs" / "random_forest.joblib").exists()
    assert (tmp_path / "outputs" / "ridge.joblib").is_file()
    assert (tmp_path / "outputs" / "mlp.joblib").is_file()
    assert "random_forest" in paths.metrics_csv.read_text(encoding="utf-8")
