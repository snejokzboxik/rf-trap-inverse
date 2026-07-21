"""Focused tests for the shared-subset v1/v2 closed-loop comparison."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from rf_trap_forward.closed_loop_comparison import (
    PER_SAMPLE_COLUMNS,
    SUMMARY_COLUMNS,
    ClosedLoopComparison,
    load_comparison_models,
    prediction_range_audit,
    run_closed_loop_comparison,
    write_comparison_outputs,
)
from rf_trap_forward.closed_loop_inverse import ClosedLoopSelection
from rf_trap_forward.inverse_training import InverseDataset, load_inverse_dataset
from rf_trap_forward.synthetic_dataset import SyntheticSolveResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConstantModel:
    def __init__(self, values_m: np.ndarray) -> None:
        self.values_m = np.asarray(values_m, dtype=float)

    def predict(self, X_m: np.ndarray) -> np.ndarray:
        return np.tile(self.values_m, (np.asarray(X_m).shape[0], 1))


def _dataset() -> InverseDataset:
    return InverseDataset(
        sample_ids=np.asarray((101, 102), dtype=np.int64),
        X_m=np.asarray(((1e-3, 0.0, 0.0, 1e-3, -1e-3, 0.0), (1.1e-3, 0.0, 0.0, 1.1e-3, -1.1e-3, 0.0))),
        y_m=np.zeros((2, 8)),
    )


@pytest.mark.filterwarnings("ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning")
def test_both_saved_models_load_and_predict_n_by_eight() -> None:
    dataset = load_inverse_dataset(PROJECT_ROOT / "validation_results/generated_dataset/synthetic_clean.csv")
    v1, v2 = load_comparison_models(
        PROJECT_ROOT / "validation_results/inverse_model_baseline/mlp.joblib",
        PROJECT_ROOT / "validation_results/inverse_model_v2/best_model.joblib",
    )
    for model in (v1, v2):
        raw, reported, _ = prediction_range_audit(model, dataset.X_m[:2])
        assert raw.shape == reported.shape == (2, 8)


def test_comparison_preserves_sample_ids_and_computes_metrics(tmp_path: Path) -> None:
    dataset = _dataset()
    selection = ClosedLoopSelection((101, 102), "mocked shared selection")

    def worker(*_: object) -> SyntheticSolveResult:
        # Each truth set shifted 10 um in x, in a different order for Hungarian matching.
        points = dataset.X_m[0].reshape(3, 2) + np.asarray((10e-6, 0.0))
        return SyntheticSolveResult(
            minima_positions_m=points[[2, 0, 1]], accepted_candidate_count=3,
            rejected_candidate_count=0, total_candidate_count=3,
            selected_interpolation_sensitive_count=0, node_count=10,
            triangle_count=12, relative_free_residual=1e-12, runtime_seconds=0.01,
        )

    comparison = run_closed_loop_comparison(
        dataset, selection=selection, v1_model=ConstantModel(np.zeros(8)),
        v2_model=ConstantModel(np.zeros(8)), batch_size=1, worker=worker,
    )
    assert isinstance(comparison, ClosedLoopComparison)
    assert all(result.report.selection.sample_ids == selection.sample_ids for result in comparison.results)
    for result in comparison.results:
        assert result.report.summary().exactly_three_count == 2
        # First row is exactly 10 um; second is intentionally not included in this mocked worker.
        assert result.report.records[0].errors_um().shape == (3,)
        assert np.allclose(result.report.records[0].errors_um(), 10.0)
    paths = write_comparison_outputs(comparison, tmp_path / "comparison")
    with paths[0].open(encoding="utf-8", newline="") as stream:
        assert tuple(csv.DictReader(stream).fieldnames or ()) == SUMMARY_COLUMNS
    with paths[1].open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream)); assert tuple(rows[0]) == PER_SAMPLE_COLUMNS
    assert len(rows) == 4 and paths[2].is_file()
