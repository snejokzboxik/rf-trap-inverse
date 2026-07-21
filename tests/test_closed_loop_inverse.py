"""Focused tests for saved-inverse forward-FEM loop closure."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from rf_trap_forward.closed_loop_inverse import (
    CLOSED_LOOP_COLUMNS,
    ClosedLoopSelection,
    closed_loop_assignment_errors_um,
    load_inverse_model,
    predict_wolfram_displacements_m,
    prepare_predicted_displacements,
    run_closed_loop_validation,
    write_closed_loop_outputs,
)
from rf_trap_forward.inverse_training import InverseDataset, load_inverse_dataset
from rf_trap_forward.synthetic_dataset import SyntheticSolveResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConstantModel:
    """Small joblib-compatible predictor used without any model training."""

    def __init__(self, prediction_m: np.ndarray) -> None:
        self.prediction_m = np.asarray(prediction_m, dtype=float)

    def predict(self, X_m: np.ndarray) -> np.ndarray:
        """Repeat the configured eight-coordinate vector for every input row."""

        return np.tile(self.prediction_m, (np.asarray(X_m).shape[0], 1))


def _small_dataset() -> InverseDataset:
    minima = np.asarray(
        (
            (1.0e-3, 2.0e-3, -2.0e-3, 0.0, 1.0e-3, -2.0e-3),
            (1.2e-3, 2.1e-3, -2.1e-3, 0.1e-3, 0.9e-3, -2.1e-3),
        )
    )
    return InverseDataset(
        sample_ids=np.asarray((11, 12), dtype=np.int64),
        X_m=minima,
        y_m=np.zeros((2, 8), dtype=float),
    )


@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning"
)
def test_saved_best_model_loads() -> None:
    """The persisted best baseline must expose a working prediction interface."""

    model = load_inverse_model(
        PROJECT_ROOT / "validation_results/inverse_model_baseline/mlp.joblib"
    )
    assert callable(model.predict)


def test_predicted_output_has_eight_wolfram_coordinates() -> None:
    """One three-minimum input must produce one flat eight-coordinate output."""

    model = ConstantModel(np.arange(8, dtype=float) * 1.0e-6)
    prediction = predict_wolfram_displacements_m(
        model,
        np.zeros((3, 2), dtype=float),
    )
    assert prediction.shape == (8,)
    assert np.array_equal(prediction, np.arange(8, dtype=float) * 1.0e-6)


def test_closed_loop_applies_wolfram_signflip_and_reorder() -> None:
    """Prepared FEM displacements must be exactly -[W3,W1,W4,W2]."""

    raw = np.asarray(((1, 2), (3, 4), (5, 6), (7, 8)), dtype=float) * 1.0e-6
    prepared = prepare_predicted_displacements(
        ConstantModel(raw.ravel()),
        np.zeros((3, 2), dtype=float),
    )
    expected = -raw[[2, 0, 3, 1]]
    assert np.array_equal(prepared.wolfram_displacements_m, raw)
    assert np.array_equal(prepared.fem_displacements_m, expected)


def test_hungarian_matching_returns_micrometre_errors() -> None:
    """Spatial assignment must undo a permutation and report distances in µm."""

    truth = np.asarray(((0.0, 0.0), (1.0e-3, 0.0), (0.0, 1.0e-3)))
    recomputed = truth[[2, 0, 1]] + np.asarray((10.0e-6, 0.0))
    matches, errors_um = closed_loop_assignment_errors_um(truth, recomputed)
    assert tuple(item.reference_index for item in matches) == (1, 2, 3)
    assert tuple(item.computed_index for item in matches) == (2, 3, 1)
    assert np.allclose(errors_um, 10.0)


def test_tiny_closed_loop_smoke_uses_mocked_forward_path(tmp_path: Path) -> None:
    """One mocked robust solve must complete and write the requested artifacts."""

    dataset = _small_dataset()
    true_minima = dataset.X_m[0].reshape(3, 2)

    def worker(*_: object) -> SyntheticSolveResult:
        return SyntheticSolveResult(
            minima_positions_m=true_minima[[1, 2, 0]],
            accepted_candidate_count=3,
            rejected_candidate_count=2,
            total_candidate_count=5,
            selected_interpolation_sensitive_count=0,
            node_count=120,
            triangle_count=210,
            relative_free_residual=1.0e-14,
            runtime_seconds=0.05,
        )

    report = run_closed_loop_validation(
        dataset,
        ConstantModel(np.zeros(8, dtype=float)),
        ClosedLoopSelection((11,), "mocked test selection"),
        batch_size=1,
        worker=worker,
    )
    summary = report.summary()
    assert summary.requested_samples == 1
    assert summary.exactly_three_count == 1
    assert summary.solver_failure_count == 0
    assert summary.ambiguous_rejected_count == 0
    assert np.isclose(summary.mean_error_um, 0.0)
    paths = write_closed_loop_outputs(report, tmp_path / "closed-loop")
    with paths.results_csv.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames or ()) == CLOSED_LOOP_COLUMNS
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
    assert paths.readme_markdown.is_file()
    summary_record = json.loads(paths.summary_json.read_text(encoding="utf-8"))
    assert summary_record["matched_minima"] == 3


@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning"
)
def test_saved_model_predicts_shape_eight_on_clean_dataset() -> None:
    """The persisted MLP and production loader must agree on the 6-to-8 schema."""

    dataset = load_inverse_dataset(
        PROJECT_ROOT / "validation_results/generated_dataset/synthetic_clean.csv"
    )
    model = load_inverse_model(
        PROJECT_ROOT / "validation_results/inverse_model_baseline/mlp.joblib"
    )
    prediction = predict_wolfram_displacements_m(model, dataset.X_m[0])
    assert prediction.shape == (8,)
    assert np.all(np.isfinite(prediction))
