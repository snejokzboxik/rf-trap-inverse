"""Export true displacements, minima, and saved-model inverse predictions."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from numpy.typing import NDArray

from .inverse_training import (
    INPUT_COLUMNS,
    MICROMETRES_PER_METRE,
    TARGET_COLUMNS,
    InverseDataset,
    load_inverse_dataset,
)


TRUE_METRE_COLUMNS = tuple(f"true_{column}" for column in TARGET_COLUMNS)
PREDICTED_METRE_COLUMNS = tuple(
    f"pred_{column}" for column in TARGET_COLUMNS
)
TRUE_MICROMETRE_COLUMNS = tuple(
    f"true_{column.removesuffix('_m')}_um" for column in TARGET_COLUMNS
)
PREDICTED_MICROMETRE_COLUMNS = tuple(
    f"pred_{column.removesuffix('_m')}_um" for column in TARGET_COLUMNS
)
ERROR_MICROMETRE_COLUMNS = tuple(
    f"error_{column.removesuffix('_m')}_um" for column in TARGET_COLUMNS
)
VECTOR_ERROR_COLUMNS = tuple(
    f"w{electrode}_vector_error_um" for electrode in range(1, 5)
)
PREDICTION_EXPORT_COLUMNS = (
    "sample_id",
    *TRUE_METRE_COLUMNS,
    *INPUT_COLUMNS,
    *PREDICTED_METRE_COLUMNS,
    *TRUE_MICROMETRE_COLUMNS,
    *PREDICTED_MICROMETRE_COLUMNS,
    *ERROR_MICROMETRE_COLUMNS,
    *VECTOR_ERROR_COLUMNS,
    "row_mae_um",
)


@dataclass(frozen=True)
class PredictionExport:
    """One deterministic subset and its inverse-model predictions."""

    sample_ids: NDArray[np.int64]
    minima_m: NDArray[np.float64]
    true_displacements_m: NDArray[np.float64]
    predicted_displacements_m: NDArray[np.float64]
    error_um: NDArray[np.float64]
    vector_error_um: NDArray[np.float64]
    row_mae_um: NDArray[np.float64]
    random_state: int

    def __post_init__(self) -> None:
        count = self.sample_ids.shape[0]
        expected = {
            "minima_m": (count, 6),
            "true_displacements_m": (count, 8),
            "predicted_displacements_m": (count, 8),
            "error_um": (count, 8),
            "vector_error_um": (count, 4),
            "row_mae_um": (count,),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True)
class PredictionExportPaths:
    """Files produced beside the requested export CSV."""

    csv: Path
    readme: Path
    summary_json: Path


def load_prediction_model(path: str | Path) -> object:
    """Load a trusted joblib estimator and require a callable predict method."""

    model = joblib.load(Path(path))
    if not callable(getattr(model, "predict", None)):
        raise ValueError("loaded inverse model does not provide predict")
    return model


def build_prediction_export(
    dataset: InverseDataset,
    model: object,
    *,
    n: int = 300,
    random_state: int = 42,
) -> PredictionExport:
    """Select a reproducible subset and calculate model-only inverse predictions."""

    row_count = dataset.X_m.shape[0]
    if n < 1:
        raise ValueError("n must be positive")
    if n > row_count:
        raise ValueError(f"n={n} exceeds available clean rows={row_count}")
    generator = np.random.default_rng(random_state)
    indices = np.sort(generator.choice(row_count, size=n, replace=False))
    minima_m = dataset.X_m[indices]
    truth_m = dataset.y_m[indices]
    prediction_m = np.asarray(model.predict(minima_m), dtype=float)
    if prediction_m.shape != truth_m.shape:
        raise ValueError(
            f"model predictions must have shape {truth_m.shape}, got {prediction_m.shape}"
        )
    if not np.all(np.isfinite(prediction_m)):
        raise ValueError("model predictions contain NaN or infinite values")
    error_um = MICROMETRES_PER_METRE * (prediction_m - truth_m)
    vector_error_um = np.linalg.norm(error_um.reshape(-1, 4, 2), axis=2)
    row_mae_um = np.mean(np.abs(error_um), axis=1)
    return PredictionExport(
        sample_ids=dataset.sample_ids[indices],
        minima_m=minima_m,
        true_displacements_m=truth_m,
        predicted_displacements_m=prediction_m,
        error_um=error_um,
        vector_error_um=vector_error_um,
        row_mae_um=row_mae_um,
        random_state=random_state,
    )


def write_prediction_export(
    export: PredictionExport,
    output_csv: str | Path,
    *,
    dataset_path: str | Path,
    model_path: str | Path,
) -> PredictionExportPaths:
    """Write the requested flat CSV plus explanatory Markdown and JSON summary."""

    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem
    paths = PredictionExportPaths(
        csv=csv_path,
        readme=csv_path.with_name(f"{stem}_readme.md"),
        summary_json=csv_path.with_name(f"{stem}_summary.json"),
    )
    with paths.csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(PREDICTION_EXPORT_COLUMNS))
        writer.writeheader()
        writer.writerows(_export_rows(export))
    summary = _summary(export, dataset_path=dataset_path, model_path=model_path)
    paths.summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    paths.readme.write_text(_readme(summary), encoding="utf-8")
    return paths


def _export_rows(export: PredictionExport) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    true_um = MICROMETRES_PER_METRE * export.true_displacements_m
    predicted_um = MICROMETRES_PER_METRE * export.predicted_displacements_m
    for row_index, sample_id in enumerate(export.sample_ids):
        row: dict[str, float | int] = {"sample_id": int(sample_id)}
        row.update(
            zip(TRUE_METRE_COLUMNS, export.true_displacements_m[row_index], strict=True)
        )
        row.update(zip(INPUT_COLUMNS, export.minima_m[row_index], strict=True))
        row.update(
            zip(
                PREDICTED_METRE_COLUMNS,
                export.predicted_displacements_m[row_index],
                strict=True,
            )
        )
        row.update(zip(TRUE_MICROMETRE_COLUMNS, true_um[row_index], strict=True))
        row.update(
            zip(PREDICTED_MICROMETRE_COLUMNS, predicted_um[row_index], strict=True)
        )
        row.update(zip(ERROR_MICROMETRE_COLUMNS, export.error_um[row_index], strict=True))
        row.update(zip(VECTOR_ERROR_COLUMNS, export.vector_error_um[row_index], strict=True))
        row["row_mae_um"] = float(export.row_mae_um[row_index])
        rows.append(row)
    return rows


def _summary(
    export: PredictionExport,
    *,
    dataset_path: str | Path,
    model_path: str | Path,
) -> dict[str, object]:
    absolute_error_um = np.abs(export.error_um)
    vectors_um = export.vector_error_um.ravel()
    return {
        "coordinate_error_sign": "predicted minus true",
        "coordinate_mae_um": float(np.mean(absolute_error_um)),
        "coordinate_max_absolute_error_um": float(np.max(absolute_error_um)),
        "coordinate_rmse_um": float(np.sqrt(np.mean(np.square(export.error_um)))),
        "dataset_path": str(Path(dataset_path)),
        "electrode_order": [
            "W1 upper-right",
            "W2 lower-right",
            "W3 upper-left",
            "W4 lower-left",
        ],
        "electrode_vector_error_max_um": float(np.max(vectors_um)),
        "electrode_vector_error_mean_um": float(np.mean(vectors_um)),
        "electrode_vector_error_median_um": float(np.median(vectors_um)),
        "electrode_vector_error_p95_um": float(np.percentile(vectors_um, 95.0)),
        "evaluation_scope": "random source rows; not guaranteed held-out",
        "finite_values": True,
        "inference_only": True,
        "model_path": str(Path(model_path)),
        "model_context": _model_context(dataset_path, model_path),
        "predicted_coordinates_outside_500_um": int(
            np.count_nonzero(np.abs(export.predicted_displacements_m) > 500.0e-6)
        ),
        "random_state": export.random_state,
        "row_count": int(export.sample_ids.size),
        "row_mae_max_um": float(np.max(export.row_mae_um)),
        "row_mae_mean_um": float(np.mean(export.row_mae_um)),
        "row_mae_median_um": float(np.median(export.row_mae_um)),
        "row_mae_p95_um": float(np.percentile(export.row_mae_um, 95.0)),
        "selection": "uniform sample without replacement, sorted by source row index",
        "wolfram_to_fem_transform": "[-W3, -W1, -W4, -W2]",
    }


def _model_context(dataset_path: str | Path, model_path: str | Path) -> str:
    """Describe the documented role of known merged-model export artifacts."""

    combined = f"{dataset_path} {model_path}".lower()
    if "merged_51974" in combined:
        return (
            "latest/largest trained pipeline and best ordinary regression MAE; "
            "merged N=29995 retains the best observed closed-loop headline metric"
        )
    if "merged_29995" in combined:
        return (
            "best observed closed-loop headline metric; merged N=51974 is the "
            "latest/largest trained pipeline and has the best ordinary regression MAE"
        )
    return "saved inverse-model inference; no ranked project role assigned"


def _readme(summary: dict[str, object]) -> str:
    return "\n".join(
        (
            "# Prediction export from a saved merged inverse model",
            "",
            f"This file contains **{summary['row_count']}** deterministic rows selected "
            f"with random state `{summary['random_state']}`. It is saved-model inference "
            "only: no new FEM solve, synthetic generation, calibration, or training was run.",
            "The rows are sampled from the full source dataset and are not guaranteed to "
            "belong to the model's held-out test split; the error summary is descriptive.",
            "",
            "## Column blocks",
            "",
            "1. `sample_id`: original row ID in the source ML dataset.",
            "2. `true_w*_d*_m`: eight true electrode-displacement coordinates in metres.",
            "3. `min*_x_m`, `min*_y_m`: six equilibrium/minimum-position coordinates in metres.",
            "4. `pred_w*_d*_m`: eight inverse-model predicted displacements in metres.",
            "5. `true_*_um` and `pred_*_um`: readable micrometre copies of the displacements.",
            "6. `error_*_um`: signed prediction error, predicted minus true, in micrometres.",
            "7. `w*_vector_error_um`: Euclidean `(dx,dy)` error for each electrode.",
            "8. `row_mae_um`: mean absolute error across all eight displacement coordinates.",
            "",
            "## Convention and provenance",
            "",
            "Raw displacement columns use Wolfram electrode order: W1 upper-right, "
            "W2 lower-right, W3 upper-left, W4 lower-left. The project transform to "
            "internal FEM order remains `[-W3, -W1, -W4, -W2]`.",
            "",
            f"- Source dataset: `{summary['dataset_path']}`",
            f"- Saved model: `{summary['model_path']}`",
            f"- Model context: {summary['model_context']}.",
            "- Metre-valued columns end in `_m`; micrometre-valued columns end in `_um`.",
            "",
            "## Inference summary",
            "",
            f"- Coordinate MAE: **{summary['coordinate_mae_um']:.6f} µm**",
            f"- Coordinate RMSE: **{summary['coordinate_rmse_um']:.6f} µm**",
            f"- Maximum absolute coordinate error: **{summary['coordinate_max_absolute_error_um']:.6f} µm**",
            f"- Mean electrode-vector error: **{summary['electrode_vector_error_mean_um']:.6f} µm**",
            f"- Mean row MAE: **{summary['row_mae_mean_um']:.6f} µm**",
            "",
        )
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the saved-model prediction-export command line."""

    parser = argparse.ArgumentParser(
        prog="rf-trap-export-predictions",
        description="Export true displacements, minima, and saved-model predictions.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "validation_results/generated_dataset_merged_51974/synthetic_clean_ml.csv"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("validation_results/inverse_model_merged_51974/mlp.joblib"),
    )
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "validation_results/prediction_export_merged_51974/"
            "prediction_dataset_300.csv"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load existing artifacts, run inference, and write the export package."""

    arguments = build_parser().parse_args(argv)
    dataset = load_inverse_dataset(arguments.dataset)
    model = load_prediction_model(arguments.model)
    export = build_prediction_export(
        dataset,
        model,
        n=arguments.n,
        random_state=arguments.random_state,
    )
    paths = write_prediction_export(
        export,
        arguments.output_csv,
        dataset_path=arguments.dataset,
        model_path=arguments.model,
    )
    summary = _summary(
        export, dataset_path=arguments.dataset, model_path=arguments.model
    )
    print(f"rows={summary['row_count']}")
    print(f"coordinate_mae_um={summary['coordinate_mae_um']}")
    print(f"coordinate_rmse_um={summary['coordinate_rmse_um']}")
    print(
        "coordinate_max_absolute_error_um="
        f"{summary['coordinate_max_absolute_error_um']}"
    )
    print(f"csv={paths.csv}")
    print(f"readme={paths.readme}")
    print(f"summary={paths.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
