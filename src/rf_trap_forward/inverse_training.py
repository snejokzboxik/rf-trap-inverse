"""Baseline inverse-model training from the accepted synthetic CSV only."""

from __future__ import annotations

import argparse
import csv
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib
import numpy as np
from numpy.typing import NDArray
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


INPUT_COLUMNS = (
    "min1_x_m",
    "min1_y_m",
    "min2_x_m",
    "min2_y_m",
    "min3_x_m",
    "min3_y_m",
)
TARGET_COLUMNS = (
    "w1_dx_m",
    "w1_dy_m",
    "w2_dx_m",
    "w2_dy_m",
    "w3_dx_m",
    "w3_dy_m",
    "w4_dx_m",
    "w4_dy_m",
)
MODEL_NAMES = ("ridge", "random_forest", "mlp")
MICROMETRES_PER_METRE = 1.0e6

METRICS_COLUMNS = (
    "model",
    "train_samples",
    "test_samples",
    "overall_mae_um",
    "overall_rmse_um",
    "max_absolute_error_um",
    "mean_electrode_vector_error_um",
    "max_electrode_vector_error_um",
    "fit_runtime_seconds",
)
PER_OUTPUT_COLUMNS = (
    "model",
    "scope",
    "coordinate",
    "electrode",
    "component",
    "mae_um",
    "rmse_um",
    "max_absolute_error_um",
    "vector_error_mean_um",
    "vector_error_max_um",
)


@dataclass(frozen=True)
class InverseDataset:
    """Six minimum coordinates and eight Wolfram-order target coordinates."""

    sample_ids: NDArray[np.int64]
    X_m: NDArray[np.float64]
    y_m: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate and copy the loaded numerical arrays."""

        sample_ids = np.asarray(self.sample_ids, dtype=np.int64)
        X_m = np.asarray(self.X_m, dtype=float)
        y_m = np.asarray(self.y_m, dtype=float)
        if X_m.ndim != 2 or X_m.shape[1] != len(INPUT_COLUMNS):
            raise ValueError("X_m must have shape (N, 6)")
        if y_m.ndim != 2 or y_m.shape[1] != len(TARGET_COLUMNS):
            raise ValueError("y_m must have shape (N, 8)")
        if sample_ids.shape != (X_m.shape[0],) or y_m.shape[0] != X_m.shape[0]:
            raise ValueError("sample IDs, X_m, and y_m must have the same row count")
        if X_m.shape[0] < 2:
            raise ValueError("at least two clean samples are required")
        if len(np.unique(sample_ids)) != sample_ids.size:
            raise ValueError("sample_id values must be unique")
        if not np.all(np.isfinite(X_m)) or not np.all(np.isfinite(y_m)):
            raise ValueError("inverse-model inputs and targets must be finite")
        object.__setattr__(self, "sample_ids", sample_ids.copy())
        object.__setattr__(self, "X_m", X_m.copy())
        object.__setattr__(self, "y_m", y_m.copy())


@dataclass(frozen=True)
class InverseSplit:
    """Deterministic train/test partition including original sample IDs."""

    train_indices: NDArray[np.int64]
    test_indices: NDArray[np.int64]
    train_sample_ids: NDArray[np.int64]
    test_sample_ids: NDArray[np.int64]
    X_train_m: NDArray[np.float64]
    X_test_m: NDArray[np.float64]
    y_train_m: NDArray[np.float64]
    y_test_m: NDArray[np.float64]


@dataclass(frozen=True)
class InverseMetrics:
    """Coordinate and electrode-vector test errors, all reported in micrometres."""

    overall_mae_um: float
    overall_rmse_um: float
    max_absolute_error_um: float
    per_output_mae_um: NDArray[np.float64]
    per_output_rmse_um: NDArray[np.float64]
    per_output_max_absolute_error_um: NDArray[np.float64]
    per_electrode_vector_error_mean_um: NDArray[np.float64]
    per_electrode_vector_error_max_um: NDArray[np.float64]


@dataclass(frozen=True)
class ModelEvaluation:
    """Fitted estimator, held-out predictions, metrics, and fit time."""

    name: str
    estimator: object
    predictions_m: NDArray[np.float64]
    metrics: InverseMetrics
    fit_runtime_seconds: float


@dataclass(frozen=True)
class InverseTrainingResult:
    """Complete deterministic baseline comparison."""

    dataset: InverseDataset
    split: InverseSplit
    evaluations: tuple[ModelEvaluation, ...]
    random_state: int
    test_size: float
    mean_predictor_metrics: InverseMetrics

    @property
    def best_evaluation(self) -> ModelEvaluation:
        """Return the model with the smallest held-out coordinate MAE."""

        return min(self.evaluations, key=lambda item: item.metrics.overall_mae_um)


@dataclass(frozen=True)
class InverseTrainingOutputPaths:
    """Files written by one inverse-training run."""

    metrics_csv: Path
    per_output_metrics_csv: Path
    test_predictions_csv: Path
    readme_markdown: Path
    plot_directory: Path
    model_paths: tuple[Path, ...]


ProgressCallback = Callable[[str, str, float | None], None]


def load_inverse_dataset(path: str | Path) -> InverseDataset:
    """Load clean minima inputs and raw Wolfram-order displacement targets.

    The loader deliberately ignores FEM-order displacement columns. Every row must
    be marked ``clean`` when a status column is present.
    """

    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("inverse dataset is missing a CSV header")
        required = {"sample_id", *INPUT_COLUMNS, *TARGET_COLUMNS}
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"inverse dataset is missing columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError("inverse dataset contains no rows")
    sample_ids: list[int] = []
    inputs: list[list[float]] = []
    targets: list[list[float]] = []
    for row_number, row in enumerate(rows, start=2):
        if "status" in row and row["status"] != "clean":
            raise ValueError(f"row {row_number} is not marked clean")
        try:
            sample_ids.append(int(row["sample_id"]))
            inputs.append([float(row[column]) for column in INPUT_COLUMNS])
            targets.append([float(row[column]) for column in TARGET_COLUMNS])
        except (TypeError, ValueError) as error:
            raise ValueError(f"malformed numeric value on CSV row {row_number}") from error
    return InverseDataset(
        sample_ids=np.asarray(sample_ids, dtype=np.int64),
        X_m=np.asarray(inputs, dtype=float),
        y_m=np.asarray(targets, dtype=float),
    )


def subset_inverse_dataset(dataset: InverseDataset, maximum_rows: int | None) -> InverseDataset:
    """Return the first requested rows for an explicit smoke run."""

    if maximum_rows is None:
        return dataset
    if maximum_rows < 2:
        raise ValueError("maximum_rows must be at least 2")
    count = min(maximum_rows, dataset.X_m.shape[0])
    return InverseDataset(
        sample_ids=dataset.sample_ids[:count],
        X_m=dataset.X_m[:count],
        y_m=dataset.y_m[:count],
    )


def split_inverse_dataset(
    dataset: InverseDataset,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> InverseSplit:
    """Create the requested reproducible random train/test split."""

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must lie strictly between zero and one")
    indices = np.arange(dataset.X_m.shape[0], dtype=np.int64)
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    return InverseSplit(
        train_indices=train_indices,
        test_indices=test_indices,
        train_sample_ids=dataset.sample_ids[train_indices],
        test_sample_ids=dataset.sample_ids[test_indices],
        X_train_m=dataset.X_m[train_indices],
        X_test_m=dataset.X_m[test_indices],
        y_train_m=dataset.y_m[train_indices],
        y_test_m=dataset.y_m[test_indices],
    )


def compute_inverse_metrics(
    true_displacements_m: NDArray[np.float64],
    predicted_displacements_m: NDArray[np.float64],
) -> InverseMetrics:
    """Compute coordinate and two-component vector errors in micrometres."""

    truth = np.asarray(true_displacements_m, dtype=float)
    prediction = np.asarray(predicted_displacements_m, dtype=float)
    if truth.ndim != 2 or truth.shape[1] != len(TARGET_COLUMNS):
        raise ValueError("true_displacements_m must have shape (N, 8)")
    if prediction.shape != truth.shape:
        raise ValueError("predicted_displacements_m must match the truth shape")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(prediction)):
        raise ValueError("metric inputs must be finite")
    error_um = MICROMETRES_PER_METRE * (prediction - truth)
    absolute_um = np.abs(error_um)
    electrode_vectors_um = np.linalg.norm(error_um.reshape(-1, 4, 2), axis=2)
    return InverseMetrics(
        overall_mae_um=float(np.mean(absolute_um)),
        overall_rmse_um=float(np.sqrt(np.mean(np.square(error_um)))),
        max_absolute_error_um=float(np.max(absolute_um)),
        per_output_mae_um=np.mean(absolute_um, axis=0),
        per_output_rmse_um=np.sqrt(np.mean(np.square(error_um), axis=0)),
        per_output_max_absolute_error_um=np.max(absolute_um, axis=0),
        per_electrode_vector_error_mean_um=np.mean(electrode_vectors_um, axis=0),
        per_electrode_vector_error_max_um=np.max(electrode_vectors_um, axis=0),
    )


def build_baseline_estimators(
    *,
    random_state: int = 42,
    smoke_test: bool = False,
) -> Mapping[str, object]:
    """Build deterministic Ridge, random-forest, and standardized MLP baselines."""

    forest_trees = 12 if smoke_test else 300
    mlp_layers = (16,) if smoke_test else (128, 64)
    mlp_iterations = 250 if smoke_test else 3000
    mlp = MLPRegressor(
        hidden_layer_sizes=mlp_layers,
        activation="relu",
        solver="lbfgs" if smoke_test else "adam",
        alpha=1.0e-4,
        batch_size="auto" if smoke_test else 32,
        learning_rate_init=1.0e-3,
        max_iter=mlp_iterations,
        tol=1.0e-3 if smoke_test else 1.0e-4,
        early_stopping=not smoke_test,
        validation_fraction=0.15,
        n_iter_no_change=100,
        random_state=random_state,
    )
    return {
        "ridge": TransformedTargetRegressor(
            regressor=Pipeline(
                (("x_scaler", StandardScaler()), ("regressor", Ridge(alpha=1.0)))
            ),
            transformer=StandardScaler(),
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=forest_trees,
            random_state=random_state,
            n_jobs=1 if smoke_test else -1,
            max_features=1.0,
        ),
        "mlp": TransformedTargetRegressor(
            regressor=Pipeline((("x_scaler", StandardScaler()), ("regressor", mlp))),
            transformer=StandardScaler(),
        ),
    }


def train_inverse_baselines(
    dataset: InverseDataset,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    smoke_test: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> InverseTrainingResult:
    """Fit and compare all three requested inverse-model baselines."""

    split = split_inverse_dataset(
        dataset,
        test_size=test_size,
        random_state=random_state,
    )
    estimators = build_baseline_estimators(
        random_state=random_state,
        smoke_test=smoke_test,
    )
    evaluations: list[ModelEvaluation] = []
    for name in MODEL_NAMES:
        estimator = estimators[name]
        if progress_callback is not None:
            progress_callback(name, "started", None)
        started = time.perf_counter()
        estimator.fit(split.X_train_m, split.y_train_m)
        runtime = time.perf_counter() - started
        predictions = np.asarray(estimator.predict(split.X_test_m), dtype=float)
        evaluation = ModelEvaluation(
            name=name,
            estimator=estimator,
            predictions_m=predictions,
            metrics=compute_inverse_metrics(split.y_test_m, predictions),
            fit_runtime_seconds=runtime,
        )
        evaluations.append(evaluation)
        if progress_callback is not None:
            progress_callback(name, "completed", runtime)
    mean_prediction = np.broadcast_to(
        np.mean(split.y_train_m, axis=0),
        split.y_test_m.shape,
    )
    return InverseTrainingResult(
        dataset=dataset,
        split=split,
        evaluations=tuple(evaluations),
        random_state=random_state,
        test_size=test_size,
        mean_predictor_metrics=compute_inverse_metrics(
            split.y_test_m,
            mean_prediction,
        ),
    )


def write_inverse_training_outputs(
    result: InverseTrainingResult,
    output_directory: str | Path,
    *,
    skip_model_artifacts: Sequence[str] = (),
) -> InverseTrainingOutputPaths:
    """Write tabular metrics, held-out predictions, plots, models, and README."""

    skipped = set(skip_model_artifacts)
    unknown = skipped.difference(MODEL_NAMES)
    if unknown:
        raise ValueError(f"unknown model artifacts requested for skipping: {sorted(unknown)}")
    output = Path(output_directory)
    plots = output / "plots"
    output.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    paths = InverseTrainingOutputPaths(
        metrics_csv=output / "metrics.csv",
        per_output_metrics_csv=output / "per_output_metrics.csv",
        test_predictions_csv=output / "test_predictions.csv",
        readme_markdown=output / "README.md",
        plot_directory=plots,
        model_paths=tuple(
            output / f"{name}.joblib" for name in MODEL_NAMES if name not in skipped
        ),
    )
    _write_metrics_csv(result, paths.metrics_csv)
    _write_per_output_csv(result, paths.per_output_metrics_csv)
    _write_predictions_csv(result, paths.test_predictions_csv)
    for evaluation in result.evaluations:
        if evaluation.name not in skipped:
            joblib.dump(evaluation.estimator, output / f"{evaluation.name}.joblib")
    _write_prediction_scatter(result, plots / "predicted_vs_true.png")
    _write_error_histogram(result, plots / "coordinate_error_histogram.png")
    _write_vector_error_boxplot(result, plots / "electrode_vector_error.png")
    paths.readme_markdown.write_text(
        _training_readme(result, tuple(name for name in MODEL_NAMES if name not in skipped)),
        encoding="utf-8",
    )
    return paths


def _write_metrics_csv(result: InverseTrainingResult, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(METRICS_COLUMNS))
        writer.writeheader()
        for evaluation in result.evaluations:
            metrics = evaluation.metrics
            writer.writerow(
                {
                    "model": evaluation.name,
                    "train_samples": result.split.X_train_m.shape[0],
                    "test_samples": result.split.X_test_m.shape[0],
                    "overall_mae_um": metrics.overall_mae_um,
                    "overall_rmse_um": metrics.overall_rmse_um,
                    "max_absolute_error_um": metrics.max_absolute_error_um,
                    "mean_electrode_vector_error_um": float(
                        np.mean(metrics.per_electrode_vector_error_mean_um)
                    ),
                    "max_electrode_vector_error_um": float(
                        np.max(metrics.per_electrode_vector_error_max_um)
                    ),
                    "fit_runtime_seconds": evaluation.fit_runtime_seconds,
                }
            )


def _write_per_output_csv(result: InverseTrainingResult, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(PER_OUTPUT_COLUMNS))
        writer.writeheader()
        for evaluation in result.evaluations:
            metrics = evaluation.metrics
            for index, coordinate in enumerate(TARGET_COLUMNS):
                electrode = index // 2 + 1
                component = "dx" if index % 2 == 0 else "dy"
                writer.writerow(
                    {
                        "model": evaluation.name,
                        "scope": "coordinate",
                        "coordinate": coordinate,
                        "electrode": f"W{electrode}",
                        "component": component,
                        "mae_um": metrics.per_output_mae_um[index],
                        "rmse_um": metrics.per_output_rmse_um[index],
                        "max_absolute_error_um": (
                            metrics.per_output_max_absolute_error_um[index]
                        ),
                        "vector_error_mean_um": "",
                        "vector_error_max_um": "",
                    }
                )
            for electrode in range(4):
                writer.writerow(
                    {
                        "model": evaluation.name,
                        "scope": "electrode_vector",
                        "coordinate": "",
                        "electrode": f"W{electrode + 1}",
                        "component": "vector_norm",
                        "mae_um": "",
                        "rmse_um": "",
                        "max_absolute_error_um": "",
                        "vector_error_mean_um": (
                            metrics.per_electrode_vector_error_mean_um[electrode]
                        ),
                        "vector_error_max_um": (
                            metrics.per_electrode_vector_error_max_um[electrode]
                        ),
                    }
                )


def _prediction_columns() -> tuple[str, ...]:
    true_columns = tuple(f"true_{column}" for column in TARGET_COLUMNS)
    predicted_columns = tuple(f"predicted_{column}" for column in TARGET_COLUMNS)
    error_columns = tuple(
        f"error_{column.removesuffix('_m')}_um" for column in TARGET_COLUMNS
    )
    vector_columns = tuple(f"w{index}_vector_error_um" for index in range(1, 5))
    return ("model", "sample_id", *true_columns, *predicted_columns, *error_columns, *vector_columns)


def _write_predictions_csv(result: InverseTrainingResult, path: Path) -> None:
    fieldnames = _prediction_columns()
    truth = result.split.y_test_m
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames))
        writer.writeheader()
        for evaluation in result.evaluations:
            error_um = MICROMETRES_PER_METRE * (evaluation.predictions_m - truth)
            vectors_um = np.linalg.norm(error_um.reshape(-1, 4, 2), axis=2)
            for row_index, sample_id in enumerate(result.split.test_sample_ids):
                row: dict[str, object] = {
                    "model": evaluation.name,
                    "sample_id": int(sample_id),
                }
                for column_index, column in enumerate(TARGET_COLUMNS):
                    row[f"true_{column}"] = truth[row_index, column_index]
                    row[f"predicted_{column}"] = evaluation.predictions_m[
                        row_index, column_index
                    ]
                    row[f"error_{column.removesuffix('_m')}_um"] = error_um[
                        row_index, column_index
                    ]
                for electrode in range(4):
                    row[f"w{electrode + 1}_vector_error_um"] = vectors_um[
                        row_index, electrode
                    ]
                writer.writerow(row)


def _write_prediction_scatter(result: InverseTrainingResult, path: Path) -> None:
    truth_um = MICROMETRES_PER_METRE * result.split.y_test_m
    combined = [truth_um]
    combined.extend(
        MICROMETRES_PER_METRE * evaluation.predictions_m
        for evaluation in result.evaluations
    )
    lower = float(min(np.min(values) for values in combined))
    upper = float(max(np.max(values) for values in combined))
    padding = 0.04 * max(upper - lower, 1.0)
    figure, axes = plt.subplots(1, 3, figsize=(14.7, 4.7), sharex=True, sharey=True)
    for axis, evaluation in zip(axes, result.evaluations, strict=True):
        prediction_um = MICROMETRES_PER_METRE * evaluation.predictions_m
        axis.scatter(truth_um.ravel(), prediction_um.ravel(), s=8, alpha=0.32)
        axis.plot(
            [lower - padding, upper + padding],
            [lower - padding, upper + padding],
            color="#991B1B",
            linewidth=1.2,
            linestyle="--",
        )
        axis.set_title(f"{evaluation.name}\nMAE {evaluation.metrics.overall_mae_um:.2f} µm")
        axis.set_xlabel("True displacement (µm)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Predicted displacement (µm)")
    axes[0].set_xlim(lower - padding, upper + padding)
    axes[0].set_ylim(lower - padding, upper + padding)
    figure.suptitle("Held-out raw Wolfram-order displacement predictions")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_error_histogram(result: InverseTrainingResult, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 5.0))
    colors = ("#2563EB", "#D97706", "#059669")
    for evaluation, color in zip(result.evaluations, colors, strict=True):
        error_um = MICROMETRES_PER_METRE * (
            evaluation.predictions_m - result.split.y_test_m
        )
        axis.hist(
            error_um.ravel(),
            bins=45,
            alpha=0.42,
            color=color,
            label=f"{evaluation.name} (MAE {evaluation.metrics.overall_mae_um:.1f} µm)",
        )
    axis.axvline(0.0, color="black", linewidth=1.0)
    axis.set(
        title="Held-out signed coordinate errors",
        xlabel="Predicted − true displacement (µm)",
        ylabel="Count",
    )
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_vector_error_boxplot(result: InverseTrainingResult, path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.7, 4.8), sharey=True)
    for axis, evaluation in zip(axes, result.evaluations, strict=True):
        error_um = MICROMETRES_PER_METRE * (
            evaluation.predictions_m - result.split.y_test_m
        )
        vectors_um = np.linalg.norm(error_um.reshape(-1, 4, 2), axis=2)
        axis.boxplot(
            [vectors_um[:, index] for index in range(4)],
            tick_labels=["W1", "W2", "W3", "W4"],
            showfliers=True,
        )
        axis.set_title(evaluation.name)
        axis.set_xlabel("Wolfram electrode")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Vector error (µm)")
    figure.suptitle("Held-out electrode displacement-vector errors")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _training_readme(
    result: InverseTrainingResult,
    saved_model_names: Sequence[str] = MODEL_NAMES,
) -> str:
    best = result.best_evaluation
    baseline = result.mean_predictor_metrics
    mae_improvement_percent = 100.0 * (
        baseline.overall_mae_um - best.metrics.overall_mae_um
    ) / baseline.overall_mae_um
    lines = [
        "# Baseline inverse-model experiment",
        "",
        "This experiment uses only `validation_results/generated_dataset/synthetic_clean.csv`. "
        "It performs no FEM solve, calibration, mesh sweep, or new data generation.",
        "",
        "## Data and convention",
        "",
        f"- Samples: {result.dataset.X_m.shape[0]} ({result.split.X_train_m.shape[0]} train, "
        f"{result.split.X_test_m.shape[0]} test).",
        f"- Split: `test_size={result.test_size}`, `random_state={result.random_state}`.",
        "- Input: three deterministic polar-angle-sorted minimum positions, shape `(N, 6)`.",
        "- Target: four raw displacement vectors in user-facing Wolfram order, shape `(N, 8)`.",
        "- Internal fit units are metres; every reported error metric is in micrometres.",
        "- Ridge and MLP standardize X and y. Random forest uses raw metre-valued features and targets.",
        "",
        "The inverse is intrinsically underdetermined at the coordinate level: six observed "
        "minimum coordinates are being used to recover eight independently sampled displacement "
        "coordinates. These baselines therefore measure useful predictive correlation; they do "
        "not establish a unique physical inverse.",
        "",
        "## Held-out metrics",
        "",
        "| Model | MAE (µm) | RMSE (µm) | Max absolute (µm) | Mean vector error (µm) | Max vector error (µm) | Fit time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for evaluation in sorted(
        result.evaluations,
        key=lambda item: item.metrics.overall_mae_um,
    ):
        metrics = evaluation.metrics
        lines.append(
            f"| {evaluation.name} | {metrics.overall_mae_um:.6f} | "
            f"{metrics.overall_rmse_um:.6f} | {metrics.max_absolute_error_um:.6f} | "
            f"{np.mean(metrics.per_electrode_vector_error_mean_um):.6f} | "
            f"{np.max(metrics.per_electrode_vector_error_max_um):.6f} | "
            f"{evaluation.fit_runtime_seconds:.3f} |"
        )
    lines.extend(
        (
            "",
            f"Best model by test MAE: **{best.name}** at "
            f"**{best.metrics.overall_mae_um:.6f} µm**.",
            f"For context, predicting the eight training-set coordinate means for every test "
            f"row gives MAE **{baseline.overall_mae_um:.6f} µm**, RMSE "
            f"**{baseline.overall_rmse_um:.6f} µm**, and maximum absolute error "
            f"**{baseline.max_absolute_error_um:.6f} µm**.",
            "",
            "## Interpretation",
            "",
            f"The best learned model reduces coordinate MAE by **{mae_improvement_percent:.2f}%** "
            "relative to the train-mean predictor. This is useful for a coarse first demo and "
            "confirms that the minima encode substantial displacement information. However, the "
            f"**{best.metrics.overall_mae_um:.2f} µm** MAE and "
            f"**{best.metrics.max_absolute_error_um:.2f} µm** worst coordinate error are not "
            "precision-control accuracy. The underdetermined 6-to-8 mapping also prevents a "
            "unique-inverse claim.",
            "",
            "`per_output_metrics.csv` contains all eight coordinate metrics and all four "
            "electrode-vector mean/max errors. `test_predictions.csv` contains one held-out "
            "prediction per model and sample. Positive coordinate error means predicted minus true.",
            "",
            "## Saved models and plots",
            "",
            "- Saved model artifacts: "
            + ", ".join(f"`{name}.joblib`" for name in saved_model_names)
            + ". Metrics still include every fitted model.",
            "- `plots/predicted_vs_true.png`",
            "- `plots/coordinate_error_histogram.png`",
            "- `plots/electrode_vector_error.png`",
            "",
            "A consumer must supply minimum coordinates in the same metre-valued, polar-angle-sorted "
            "order used by the generated dataset. Joblib files should only be loaded from this trusted project output.",
            "",
        )
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the baseline inverse-training command line."""

    parser = argparse.ArgumentParser(
        prog="rf-trap-train-inverse",
        description="Train inverse baselines from the existing clean synthetic CSV.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("validation_results/generated_dataset/synthetic_clean.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation_results/inverse_model_baseline"),
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Use only the first N rows for an explicit smoke run.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use lightweight model settings; intended only for tests.",
    )
    parser.add_argument(
        "--skip-model-artifact",
        action="append",
        choices=MODEL_NAMES,
        default=[],
        help="fit and evaluate a model but do not serialize its joblib artifact",
    )
    return parser


def _print_progress(name: str, status: str, runtime_seconds: float | None) -> None:
    if status == "started":
        print(f"training {name}...", flush=True)
    else:
        print(f"completed {name} in {runtime_seconds:.3f} s", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Train all baselines and write the requested comparison artifacts."""

    arguments = build_parser().parse_args(argv)
    dataset = subset_inverse_dataset(
        load_inverse_dataset(arguments.dataset),
        arguments.max_rows,
    )
    result = train_inverse_baselines(
        dataset,
        test_size=arguments.test_size,
        random_state=arguments.random_state,
        smoke_test=arguments.smoke_test,
        progress_callback=_print_progress,
    )
    paths = write_inverse_training_outputs(
        result,
        arguments.output_dir,
        skip_model_artifacts=arguments.skip_model_artifact,
    )
    best = result.best_evaluation
    print(f"samples={dataset.X_m.shape[0]}")
    print(f"train_samples={result.split.X_train_m.shape[0]}")
    print(f"test_samples={result.split.X_test_m.shape[0]}")
    print(f"best_model={best.name}")
    print(f"best_test_mae_um={best.metrics.overall_mae_um:.9g}")
    print(f"best_test_rmse_um={best.metrics.overall_rmse_um:.9g}")
    print(f"best_test_max_absolute_error_um={best.metrics.max_absolute_error_um:.9g}")
    print(f"metrics={paths.metrics_csv}")
    print(f"readme={paths.readme_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
