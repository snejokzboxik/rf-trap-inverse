"""Tests for the improved inverse-model comparison."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from rf_trap_forward.inverse_training import InverseDataset
from rf_trap_forward.inverse_model_artifacts import ClippedInverseModel
from rf_trap_forward.inverse_training_v2 import (
    DISPLACEMENT_LIMIT_M,
    MODEL_NAMES,
    clip_displacement_predictions_m,
    evaluate_prediction_variants,
    load_v2_dataset,
    train_inverse_v2,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _small_dataset(rows: int = 64) -> InverseDataset:
    generator = np.random.default_rng(17)
    X_m = generator.normal(scale=2.0e-3, size=(rows, 6))
    coefficients = generator.normal(scale=0.07, size=(6, 8))
    y_m = X_m @ coefficients
    return InverseDataset(
        sample_ids=np.arange(1, rows + 1, dtype=np.int64),
        X_m=X_m,
        y_m=y_m,
    )


def test_v2_dataset_loading_uses_expected_shapes() -> None:
    """The existing clean dataset must load as six inputs and eight W targets."""

    dataset = load_v2_dataset(
        PROJECT_ROOT / "validation_results/generated_dataset/synthetic_clean.csv"
    )
    assert dataset.X_m.shape == (1000, 6)
    assert dataset.y_m.shape == (1000, 8)


def test_clipping_is_coordinatewise_and_bounded() -> None:
    """Only predictions beyond ±500 µm may be changed by clipping."""

    raw = np.asarray(
        ((-700.0e-6, -500.0e-6, -20.0e-6, 0.0, 20.0e-6, 500.0e-6, 510.0e-6, 900.0e-6),)
    )
    clipped = clip_displacement_predictions_m(raw)
    expected = np.asarray(
        ((-500.0e-6, -500.0e-6, -20.0e-6, 0.0, 20.0e-6, 500.0e-6, 500.0e-6, 500.0e-6),)
    )
    assert np.array_equal(clipped, expected)
    assert np.max(np.abs(clipped)) == DISPLACEMENT_LIMIT_M


def test_v2_metric_units_are_micrometres() -> None:
    """A two-micrometre raw coordinate error must be reported as two."""

    truth = np.zeros((2, 8), dtype=float)
    predictions = np.full((2, 8), 2.0e-6, dtype=float)
    evaluation = evaluate_prediction_variants(
        "dummy",
        object(),
        truth,
        predictions,
        0.01,
    )
    assert np.isclose(evaluation.raw.metrics.overall_mae_um, 2.0)
    assert np.isclose(evaluation.raw.metrics.overall_rmse_um, 2.0)
    assert np.isclose(evaluation.raw.metrics.max_absolute_error_um, 2.0)


def test_all_v2_model_outputs_have_eight_targets() -> None:
    """Every requested estimator must produce an `(N_test, 8)` array."""

    result = train_inverse_v2(
        _small_dataset(),
        test_size=0.2,
        random_state=42,
        repeat_count=1,
        smoke_test=True,
    )
    assert tuple(item.name for item in result.primary_evaluations) == MODEL_NAMES
    for evaluation in result.primary_evaluations:
        assert evaluation.raw.predictions_m.shape == (13, 8)
        assert evaluation.clipped.predictions_m.shape == (13, 8)
        assert np.all(np.isfinite(evaluation.raw.predictions_m))


def test_tiny_v2_training_smoke_records_raw_and_clipped_metrics() -> None:
    """A small one-split comparison must run without FEM or data generation."""

    result = train_inverse_v2(
        _small_dataset(rows=48),
        repeat_count=1,
        smoke_test=True,
    )
    assert len(result.primary_evaluations) == 4
    assert len(result.repeated_evaluations) == 8
    assert {item.prediction_variant for item in result.repeated_evaluations} == {
        "raw",
        "clipped",
    }
    assert result.best.variant_evaluation.metrics.overall_mae_um >= 0.0


@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning"
)
def test_clipped_best_model_is_joblib_reloadable(tmp_path: Path) -> None:
    """The saved clipping wrapper must use an import-stable module path."""

    X_m = np.zeros((4, 6), dtype=float)
    y_m = np.full((4, 8), 700.0e-6, dtype=float)
    estimator = DummyRegressor(strategy="mean").fit(X_m, y_m)
    path = tmp_path / "best_model.joblib"
    joblib.dump(ClippedInverseModel(estimator), path)
    restored = joblib.load(path)
    prediction = restored.predict(np.zeros((2, 6), dtype=float))
    assert prediction.shape == (2, 8)
    assert np.allclose(prediction, 500.0e-6)
