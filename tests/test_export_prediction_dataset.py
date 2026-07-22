"""Tests for the saved inverse-model prediction export."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from rf_trap_forward.export_prediction_dataset import (
    ERROR_MICROMETRE_COLUMNS,
    PREDICTED_MICROMETRE_COLUMNS,
    PREDICTION_EXPORT_COLUMNS,
    TRUE_MICROMETRE_COLUMNS,
    build_prediction_export,
    write_prediction_export,
)
from rf_trap_forward.inverse_training import InverseDataset


class _OffsetModel:
    """Return fixed, finite eight-coordinate predictions for export tests."""

    def predict(self, inputs_m: np.ndarray) -> np.ndarray:
        offsets_m = np.arange(1, 9, dtype=float) * 1.0e-6
        return np.tile(offsets_m, (inputs_m.shape[0], 1))


def _dataset(rows: int = 12) -> InverseDataset:
    minima = np.arange(rows * 6, dtype=float).reshape(rows, 6) * 1.0e-5
    truth = np.zeros((rows, 8), dtype=float)
    return InverseDataset(np.arange(1, rows + 1), minima, truth)


def test_prediction_export_schema_count_finiteness_and_units(tmp_path: Path) -> None:
    """The flat table must have the stable schema and correct µm conversions."""

    export = build_prediction_export(
        _dataset(), _OffsetModel(), n=5, random_state=17
    )
    paths = write_prediction_export(
        export,
        tmp_path / "prediction_dataset_5.csv",
        dataset_path="source.csv",
        model_path="mlp.joblib",
    )

    with paths.csv.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        assert tuple(reader.fieldnames or ()) == PREDICTION_EXPORT_COLUMNS
    assert len(rows) == 5
    numeric = np.asarray(
        [[float(row[column]) for column in PREDICTION_EXPORT_COLUMNS[1:]] for row in rows]
    )
    assert np.all(np.isfinite(numeric))
    assert export.predicted_displacements_m.shape == (5, 8)
    first = rows[0]
    assert float(first[TRUE_MICROMETRE_COLUMNS[0]]) == 0.0
    assert float(first[PREDICTED_MICROMETRE_COLUMNS[0]]) == 1.0
    assert float(first[ERROR_MICROMETRE_COLUMNS[0]]) == 1.0
    assert float(first["row_mae_um"]) == 4.5
    summary = json.loads(paths.summary_json.read_text(encoding="utf-8"))
    assert summary["row_count"] == 5
    assert summary["finite_values"] is True
    assert summary["inference_only"] is True
    assert summary["evaluation_scope"] == "random source rows; not guaranteed held-out"


def test_prediction_export_selection_is_deterministic() -> None:
    """The same random state must select the same source sample IDs."""

    first = build_prediction_export(_dataset(), _OffsetModel(), n=7, random_state=3)
    second = build_prediction_export(_dataset(), _OffsetModel(), n=7, random_state=3)
    assert np.array_equal(first.sample_ids, second.sample_ids)
    assert np.array_equal(first.predicted_displacements_m, second.predicted_displacements_m)
