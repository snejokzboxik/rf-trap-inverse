"""Improved deterministic inverse-model comparison on the existing clean CSV."""

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
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .inverse_training import (
    InverseDataset,
    InverseMetrics,
    InverseSplit,
    TARGET_COLUMNS,
    compute_inverse_metrics,
    load_inverse_dataset,
    split_inverse_dataset,
    subset_inverse_dataset,
)
from .inverse_model_artifacts import ClippedInverseModel

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


MODEL_NAMES = ("tuned_mlp", "extra_trees", "hist_gradient_boosting", "knn")
PREDICTION_VARIANTS = ("raw", "clipped")
DISPLACEMENT_LIMIT_M = 500.0e-6
BASELINE_MLP_MAE_UM = 119.15431192444434
BASELINE_MLP_MAX_ABSOLUTE_ERROR_UM = 439.3281308736103

METRICS_COLUMNS = (
    "model",
    "prediction_variant",
    "train_samples",
    "test_samples",
    "overall_mae_um",
    "overall_rmse_um",
    "max_absolute_error_um",
    "mean_electrode_vector_error_um",
    "max_electrode_vector_error_um",
    "outside_range_coordinates_before_clipping",
    "outside_range_coordinates_after_clipping",
    "fit_runtime_seconds",
)
REPEATED_METRICS_COLUMNS = (
    "split_index",
    "random_state",
    *METRICS_COLUMNS,
)
PER_OUTPUT_COLUMNS = (
    "model",
    "prediction_variant",
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
class PredictionVariantEvaluation:
    """Predictions and held-out metrics for raw or bounded output."""

    name: str
    predictions_m: NDArray[np.float64]
    metrics: InverseMetrics
    outside_range_coordinates_after_clipping: int


@dataclass(frozen=True)
class V2ModelEvaluation:
    """One fitted v2 estimator and its raw/clipped primary-split results."""

    name: str
    estimator: object
    raw: PredictionVariantEvaluation
    clipped: PredictionVariantEvaluation
    outside_range_coordinates_before_clipping: int
    fit_runtime_seconds: float

    def variant(self, name: str) -> PredictionVariantEvaluation:
        """Return the requested raw or clipped evaluation."""

        if name == "raw":
            return self.raw
        if name == "clipped":
            return self.clipped
        raise ValueError("prediction variant must be 'raw' or 'clipped'")


@dataclass(frozen=True)
class RepeatedSplitEvaluation:
    """Metrics from one model/variant on one deterministic split."""

    split_index: int
    random_state: int
    model_name: str
    prediction_variant: str
    train_samples: int
    test_samples: int
    metrics: InverseMetrics
    outside_range_coordinates_before_clipping: int
    outside_range_coordinates_after_clipping: int
    fit_runtime_seconds: float


@dataclass(frozen=True)
class BestModelSelection:
    """Lowest-MAE primary-split model and prediction variant."""

    model_evaluation: V2ModelEvaluation
    prediction_variant: str

    @property
    def variant_evaluation(self) -> PredictionVariantEvaluation:
        """Return the winning held-out prediction variant."""

        return self.model_evaluation.variant(self.prediction_variant)


@dataclass(frozen=True)
class V2TrainingResult:
    """Primary comparison plus all deterministic repeated-split metrics."""

    dataset: InverseDataset
    primary_split: InverseSplit
    primary_evaluations: tuple[V2ModelEvaluation, ...]
    repeated_evaluations: tuple[RepeatedSplitEvaluation, ...]
    random_states: tuple[int, ...]
    test_size: float

    @property
    def best(self) -> BestModelSelection:
        """Select the lowest-MAE model/variant, preferring raw on exact ties."""

        candidates = [
            BestModelSelection(model, variant)
            for model in self.primary_evaluations
            for variant in PREDICTION_VARIANTS
        ]
        return min(
            candidates,
            key=lambda item: (
                item.variant_evaluation.metrics.overall_mae_um,
                item.prediction_variant != "raw",
            ),
        )


@dataclass(frozen=True)
class V2OutputPaths:
    """Requested v2 tables, report, plots, and winning model."""

    metrics_csv: Path
    repeated_split_metrics_csv: Path
    per_output_metrics_csv: Path
    test_predictions_csv: Path
    readme_markdown: Path
    best_model_joblib: Path
    plot_directory: Path


ProgressCallback = Callable[[int, int, str, str, float | None], None]


def load_v2_dataset(path: str | Path) -> InverseDataset:
    """Load the existing clean six-input/eight-Wolfram-target dataset."""

    return load_inverse_dataset(path)


def clip_displacement_predictions_m(
    predictions_m: NDArray[np.float64],
    *,
    limit_m: float = DISPLACEMENT_LIMIT_M,
) -> NDArray[np.float64]:
    """Clip finite eight-coordinate predictions to the known training bounds."""

    predictions = np.asarray(predictions_m, dtype=float)
    if predictions.ndim != 2 or predictions.shape[1] != len(TARGET_COLUMNS):
        raise ValueError("predictions_m must have shape (N, 8)")
    if not np.all(np.isfinite(predictions)):
        raise ValueError("predictions_m must be finite")
    if not np.isfinite(limit_m) or limit_m <= 0.0:
        raise ValueError("limit_m must be finite and positive")
    return np.clip(predictions, -limit_m, limit_m)


def build_v2_estimators(
    *,
    random_state: int = 42,
    smoke_test: bool = False,
) -> Mapping[str, object]:
    """Build the four requested deterministic improved inverse estimators."""

    mlp = MLPRegressor(
        hidden_layer_sizes=(32, 16) if smoke_test else (256, 128, 64),
        activation="tanh" if not smoke_test else "relu",
        solver="lbfgs" if smoke_test else "adam",
        alpha=3.0e-4,
        batch_size="auto" if smoke_test else 32,
        learning_rate_init=3.0e-4,
        max_iter=250 if smoke_test else 4000,
        tol=1.0e-3 if smoke_test else 1.0e-5,
        early_stopping=not smoke_test,
        validation_fraction=0.15,
        n_iter_no_change=120,
        random_state=random_state,
    )
    histogram = HistGradientBoostingRegressor(
        learning_rate=0.08 if smoke_test else 0.05,
        max_iter=25 if smoke_test else 400,
        max_leaf_nodes=7 if smoke_test else 31,
        min_samples_leaf=5 if smoke_test else 15,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15 if smoke_test else 35,
        tol=1.0e-4 if smoke_test else 1.0e-7,
        random_state=random_state,
    )
    return {
        "tuned_mlp": TransformedTargetRegressor(
            regressor=Pipeline(
                (("x_scaler", StandardScaler()), ("regressor", mlp))
            ),
            transformer=StandardScaler(),
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=16 if smoke_test else 600,
            max_depth=6 if smoke_test else None,
            min_samples_leaf=1,
            max_features=1.0,
            n_jobs=1 if smoke_test else -1,
            random_state=random_state,
        ),
        "hist_gradient_boosting": TransformedTargetRegressor(
            regressor=Pipeline(
                (
                    ("x_scaler", StandardScaler()),
                    (
                        "regressor",
                        MultiOutputRegressor(histogram),
                    ),
                )
            ),
            transformer=StandardScaler(),
        ),
        "knn": Pipeline(
            (
                ("x_scaler", StandardScaler()),
                (
                    "regressor",
                    KNeighborsRegressor(
                        n_neighbors=3 if smoke_test else 10,
                        weights="distance",
                        p=2,
                    ),
                ),
            )
        ),
    }


def train_inverse_v2(
    dataset: InverseDataset,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    repeat_count: int = 5,
    smoke_test: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> V2TrainingResult:
    """Fit all v2 models on the primary and repeated deterministic splits."""

    if repeat_count <= 0:
        raise ValueError("repeat_count must be positive")
    random_states = tuple(random_state + index for index in range(repeat_count))
    primary_split: InverseSplit | None = None
    primary_evaluations: list[V2ModelEvaluation] = []
    repeated: list[RepeatedSplitEvaluation] = []
    for split_index, state in enumerate(random_states, start=1):
        split = split_inverse_dataset(
            dataset,
            test_size=test_size,
            random_state=state,
        )
        if primary_split is None:
            primary_split = split
        estimators = build_v2_estimators(
            random_state=state,
            smoke_test=smoke_test,
        )
        for model_name in MODEL_NAMES:
            estimator = estimators[model_name]
            if progress_callback is not None:
                progress_callback(split_index, repeat_count, model_name, "started", None)
            started = time.perf_counter()
            estimator.fit(split.X_train_m, split.y_train_m)
            fit_runtime = time.perf_counter() - started
            raw_predictions = np.asarray(estimator.predict(split.X_test_m), dtype=float)
            evaluation = evaluate_prediction_variants(
                model_name,
                estimator,
                split.y_test_m,
                raw_predictions,
                fit_runtime,
            )
            if split_index == 1:
                primary_evaluations.append(evaluation)
            for variant_name in PREDICTION_VARIANTS:
                variant = evaluation.variant(variant_name)
                repeated.append(
                    RepeatedSplitEvaluation(
                        split_index=split_index,
                        random_state=state,
                        model_name=model_name,
                        prediction_variant=variant_name,
                        train_samples=split.X_train_m.shape[0],
                        test_samples=split.X_test_m.shape[0],
                        metrics=variant.metrics,
                        outside_range_coordinates_before_clipping=(
                            evaluation.outside_range_coordinates_before_clipping
                        ),
                        outside_range_coordinates_after_clipping=(
                            variant.outside_range_coordinates_after_clipping
                        ),
                        fit_runtime_seconds=fit_runtime,
                    )
                )
            if progress_callback is not None:
                progress_callback(
                    split_index,
                    repeat_count,
                    model_name,
                    "completed",
                    fit_runtime,
                )
    if primary_split is None:
        raise RuntimeError("no primary split was generated")
    return V2TrainingResult(
        dataset=dataset,
        primary_split=primary_split,
        primary_evaluations=tuple(primary_evaluations),
        repeated_evaluations=tuple(repeated),
        random_states=random_states,
        test_size=test_size,
    )


def evaluate_prediction_variants(
    model_name: str,
    estimator: object,
    truth_m: NDArray[np.float64],
    raw_predictions_m: NDArray[np.float64],
    fit_runtime_seconds: float,
) -> V2ModelEvaluation:
    """Evaluate raw and coordinate-wise clipped predictions in micrometres."""

    raw = np.asarray(raw_predictions_m, dtype=float)
    truth = np.asarray(truth_m, dtype=float)
    if raw.shape != truth.shape or raw.ndim != 2 or raw.shape[1] != 8:
        raise ValueError("raw predictions and truth must have equal shape (N, 8)")
    if not np.all(np.isfinite(raw)):
        raise ValueError("raw predictions must be finite")
    clipped = clip_displacement_predictions_m(raw)
    outside_before = int(np.count_nonzero(np.abs(raw) > DISPLACEMENT_LIMIT_M))
    outside_after = int(np.count_nonzero(np.abs(clipped) > DISPLACEMENT_LIMIT_M))
    return V2ModelEvaluation(
        name=model_name,
        estimator=estimator,
        raw=PredictionVariantEvaluation(
            name="raw",
            predictions_m=raw.copy(),
            metrics=compute_inverse_metrics(truth, raw),
            outside_range_coordinates_after_clipping=outside_before,
        ),
        clipped=PredictionVariantEvaluation(
            name="clipped",
            predictions_m=clipped,
            metrics=compute_inverse_metrics(truth, clipped),
            outside_range_coordinates_after_clipping=outside_after,
        ),
        outside_range_coordinates_before_clipping=outside_before,
        fit_runtime_seconds=fit_runtime_seconds,
    )


def write_v2_outputs(
    result: V2TrainingResult,
    output_directory: str | Path,
) -> V2OutputPaths:
    """Write v2 metrics, predictions, plots, README, and winning joblib model."""

    output = Path(output_directory)
    plots = output / "plots"
    output.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    paths = V2OutputPaths(
        metrics_csv=output / "metrics.csv",
        repeated_split_metrics_csv=output / "repeated_split_metrics.csv",
        per_output_metrics_csv=output / "per_output_metrics.csv",
        test_predictions_csv=output / "test_predictions.csv",
        readme_markdown=output / "README.md",
        best_model_joblib=output / "best_model.joblib",
        plot_directory=plots,
    )
    _write_metrics(result, paths.metrics_csv)
    _write_repeated_metrics(result, paths.repeated_split_metrics_csv)
    _write_per_output_metrics(result, paths.per_output_metrics_csv)
    _write_test_predictions(result, paths.test_predictions_csv)
    best = result.best
    saved_model: object = best.model_evaluation.estimator
    if best.prediction_variant == "clipped":
        saved_model = ClippedInverseModel(saved_model)
    joblib.dump(saved_model, paths.best_model_joblib, compress=3)
    _write_prediction_scatter(result, plots / "predicted_vs_true.png")
    _write_error_histogram(result, plots / "error_histogram.png")
    _write_vector_error_plot(result, plots / "per_electrode_vector_error.png")
    _write_model_comparison(result, plots / "model_comparison.png")
    paths.readme_markdown.write_text(_training_readme(result), encoding="utf-8")
    return paths


def _metric_record(
    model_name: str,
    prediction_variant: str,
    train_samples: int,
    test_samples: int,
    metrics: InverseMetrics,
    outside_before: int,
    outside_after: int,
    fit_runtime_seconds: float,
) -> dict[str, object]:
    return {
        "model": model_name,
        "prediction_variant": prediction_variant,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "overall_mae_um": metrics.overall_mae_um,
        "overall_rmse_um": metrics.overall_rmse_um,
        "max_absolute_error_um": metrics.max_absolute_error_um,
        "mean_electrode_vector_error_um": float(
            np.mean(metrics.per_electrode_vector_error_mean_um)
        ),
        "max_electrode_vector_error_um": float(
            np.max(metrics.per_electrode_vector_error_max_um)
        ),
        "outside_range_coordinates_before_clipping": outside_before,
        "outside_range_coordinates_after_clipping": outside_after,
        "fit_runtime_seconds": fit_runtime_seconds,
    }


def _write_metrics(result: V2TrainingResult, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(METRICS_COLUMNS))
        writer.writeheader()
        for evaluation in result.primary_evaluations:
            for variant_name in PREDICTION_VARIANTS:
                variant = evaluation.variant(variant_name)
                writer.writerow(
                    _metric_record(
                        evaluation.name,
                        variant_name,
                        result.primary_split.X_train_m.shape[0],
                        result.primary_split.X_test_m.shape[0],
                        variant.metrics,
                        evaluation.outside_range_coordinates_before_clipping,
                        variant.outside_range_coordinates_after_clipping,
                        evaluation.fit_runtime_seconds,
                    )
                )


def _write_repeated_metrics(result: V2TrainingResult, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(REPEATED_METRICS_COLUMNS),
        )
        writer.writeheader()
        for evaluation in result.repeated_evaluations:
            row = _metric_record(
                evaluation.model_name,
                evaluation.prediction_variant,
                evaluation.train_samples,
                evaluation.test_samples,
                evaluation.metrics,
                evaluation.outside_range_coordinates_before_clipping,
                evaluation.outside_range_coordinates_after_clipping,
                evaluation.fit_runtime_seconds,
            )
            writer.writerow(
                {
                    "split_index": evaluation.split_index,
                    "random_state": evaluation.random_state,
                    **row,
                }
            )


def _write_per_output_metrics(result: V2TrainingResult, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(PER_OUTPUT_COLUMNS))
        writer.writeheader()
        for evaluation in result.primary_evaluations:
            for variant_name in PREDICTION_VARIANTS:
                metrics = evaluation.variant(variant_name).metrics
                for index, coordinate in enumerate(TARGET_COLUMNS):
                    writer.writerow(
                        {
                            "model": evaluation.name,
                            "prediction_variant": variant_name,
                            "scope": "coordinate",
                            "coordinate": coordinate,
                            "electrode": f"W{index // 2 + 1}",
                            "component": "dx" if index % 2 == 0 else "dy",
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
                            "prediction_variant": variant_name,
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
    prediction_columns = tuple(f"predicted_{column}" for column in TARGET_COLUMNS)
    error_columns = tuple(
        f"error_{column.removesuffix('_m')}_um" for column in TARGET_COLUMNS
    )
    vector_columns = tuple(f"w{index}_vector_error_um" for index in range(1, 5))
    return (
        "model",
        "prediction_variant",
        "sample_id",
        *true_columns,
        *prediction_columns,
        *error_columns,
        *vector_columns,
    )


def _write_test_predictions(result: V2TrainingResult, path: Path) -> None:
    fieldnames = _prediction_columns()
    truth = result.primary_split.y_test_m
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames))
        writer.writeheader()
        for evaluation in result.primary_evaluations:
            for variant_name in PREDICTION_VARIANTS:
                predictions = evaluation.variant(variant_name).predictions_m
                error_um = 1.0e6 * (predictions - truth)
                vectors_um = np.linalg.norm(error_um.reshape(-1, 4, 2), axis=2)
                for row_index, sample_id in enumerate(
                    result.primary_split.test_sample_ids
                ):
                    row: dict[str, object] = {
                        "model": evaluation.name,
                        "prediction_variant": variant_name,
                        "sample_id": int(sample_id),
                    }
                    for column_index, column in enumerate(TARGET_COLUMNS):
                        row[f"true_{column}"] = truth[row_index, column_index]
                        row[f"predicted_{column}"] = predictions[
                            row_index, column_index
                        ]
                        row[
                            f"error_{column.removesuffix('_m')}_um"
                        ] = error_um[row_index, column_index]
                    for electrode in range(4):
                        row[f"w{electrode + 1}_vector_error_um"] = vectors_um[
                            row_index, electrode
                        ]
                    writer.writerow(row)


def _write_prediction_scatter(result: V2TrainingResult, path: Path) -> None:
    truth_um = 1.0e6 * result.primary_split.y_test_m
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 9.8), sharex=True, sharey=True)
    lower = -520.0
    upper = 520.0
    for axis, evaluation in zip(axes.ravel(), result.primary_evaluations, strict=True):
        predictions_um = 1.0e6 * evaluation.clipped.predictions_m
        axis.scatter(truth_um.ravel(), predictions_um.ravel(), s=8, alpha=0.30)
        axis.plot((lower, upper), (lower, upper), "--", color="#991B1B", linewidth=1.1)
        axis.set_title(
            f"{evaluation.name}\nclipped MAE {evaluation.clipped.metrics.overall_mae_um:.2f} µm"
        )
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.grid(alpha=0.2)
    figure.supxlabel("True Wolfram displacement (µm)")
    figure.supylabel("Predicted Wolfram displacement (µm)")
    figure.suptitle("V2 held-out predictions (clipped variants)")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_error_histogram(result: V2TrainingResult, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    colors = ("#2563EB", "#D97706", "#059669", "#7C3AED")
    for evaluation, color in zip(result.primary_evaluations, colors, strict=True):
        errors_um = 1.0e6 * (
            evaluation.clipped.predictions_m - result.primary_split.y_test_m
        )
        axis.hist(
            errors_um.ravel(),
            bins=48,
            alpha=0.34,
            color=color,
            label=f"{evaluation.name} ({evaluation.clipped.metrics.overall_mae_um:.1f} µm)",
        )
    axis.axvline(0.0, color="black", linewidth=1.0)
    axis.set(
        title="V2 held-out signed coordinate errors (clipped)",
        xlabel="Predicted − true displacement (µm)",
        ylabel="Count",
    )
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_vector_error_plot(result: V2TrainingResult, path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 9.0), sharey=True)
    for axis, evaluation in zip(axes.ravel(), result.primary_evaluations, strict=True):
        error_um = 1.0e6 * (
            evaluation.clipped.predictions_m - result.primary_split.y_test_m
        )
        vectors_um = np.linalg.norm(error_um.reshape(-1, 4, 2), axis=2)
        axis.boxplot(
            [vectors_um[:, index] for index in range(4)],
            tick_labels=("W1", "W2", "W3", "W4"),
            showfliers=True,
        )
        axis.set_title(evaluation.name)
        axis.set_xlabel("Wolfram electrode")
        axis.grid(axis="y", alpha=0.2)
    figure.supylabel("Vector error (µm)")
    figure.suptitle("V2 held-out electrode-vector errors (clipped)")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_model_comparison(result: V2TrainingResult, path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15.4, 5.0))
    x = np.arange(len(result.primary_evaluations))
    width = 0.36
    metric_specs = (
        ("overall_mae_um", "MAE (µm)"),
        ("overall_rmse_um", "RMSE (µm)"),
        ("max_absolute_error_um", "Maximum absolute error (µm)"),
    )
    for axis, (attribute, label) in zip(axes, metric_specs, strict=True):
        raw = [getattr(item.raw.metrics, attribute) for item in result.primary_evaluations]
        clipped = [
            getattr(item.clipped.metrics, attribute) for item in result.primary_evaluations
        ]
        axis.bar(x - width / 2, raw, width, label="raw", color="#64748B")
        axis.bar(x + width / 2, clipped, width, label="clipped", color="#0F766E")
        axis.set_xticks(x, MODEL_NAMES, rotation=18, ha="right")
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend()
    figure.suptitle("Primary-split v2 inverse-model comparison")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _repeated_clipped_statistics(
    result: V2TrainingResult,
) -> dict[str, tuple[float, float, float, float]]:
    statistics = {}
    for model_name in MODEL_NAMES:
        values = np.asarray(
            [
                item.metrics.overall_mae_um
                for item in result.repeated_evaluations
                if item.model_name == model_name
                and item.prediction_variant == "clipped"
            ]
        )
        statistics[model_name] = (
            float(np.mean(values)),
            float(np.std(values)),
            float(np.min(values)),
            float(np.max(values)),
        )
    return statistics


def _training_readme(result: V2TrainingResult) -> str:
    best = result.best
    best_metrics = best.variant_evaluation.metrics
    improvement = BASELINE_MLP_MAE_UM - best_metrics.overall_mae_um
    maximum_error_change = (
        BASELINE_MLP_MAX_ABSOLUTE_ERROR_UM - best_metrics.max_absolute_error_um
    )
    repeated = _repeated_clipped_statistics(result)
    clipping_max_improved_models = [
        evaluation.name
        for evaluation in result.primary_evaluations
        if evaluation.clipped.metrics.max_absolute_error_um
        < evaluation.raw.metrics.max_absolute_error_um
    ]
    replacement = improvement > 0.0 and maximum_error_change >= 0.0
    lines = [
        "# Improved inverse-model comparison (v2)",
        "",
        "This experiment reads only the existing QA-passed N=1000 clean CSV. It runs no FEM solve, data generation, closed-loop validation, calibration, or mesh sweep.",
        "",
        "## Evaluation design",
        "",
        f"- Primary split: `test_size={result.test_size}`, `random_state={result.random_states[0]}`; {result.primary_split.X_train_m.shape[0]} train and {result.primary_split.X_test_m.shape[0]} test rows.",
        f"- Repeated splits: {len(result.random_states)} seeds: `{', '.join(str(value) for value in result.random_states)}`.",
        "- X: six polar-angle-sorted minimum coordinates in metres.",
        "- y: eight raw W1--W4 displacement coordinates in Wolfram order, in metres.",
        "- Reported errors are in micrometres.",
        "- Raw predictions and coordinate-wise clipping to ±500 µm are evaluated separately. The saved model includes clipping only when the clipped variant wins.",
        "",
        "## Primary-split metrics",
        "",
        "| Model | Variant | MAE (µm) | RMSE (µm) | Max absolute (µm) | Outside before/after | Fit time (s) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for evaluation in result.primary_evaluations:
        for variant_name in PREDICTION_VARIANTS:
            variant = evaluation.variant(variant_name)
            metrics = variant.metrics
            lines.append(
                f"| {evaluation.name} | {variant_name} | {metrics.overall_mae_um:.6f} | "
                f"{metrics.overall_rmse_um:.6f} | {metrics.max_absolute_error_um:.6f} | "
                f"{evaluation.outside_range_coordinates_before_clipping}/"
                f"{variant.outside_range_coordinates_after_clipping} | "
                f"{evaluation.fit_runtime_seconds:.3f} |"
            )
    lines.extend(
        (
            "",
            "## Five-split stability (clipped variants)",
            "",
            "| Model | Mean MAE (µm) | Std (µm) | Minimum (µm) | Maximum (µm) |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for model_name in MODEL_NAMES:
        mean, standard_deviation, minimum, maximum = repeated[model_name]
        lines.append(
            f"| {model_name} | {mean:.6f} | {standard_deviation:.6f} | "
            f"{minimum:.6f} | {maximum:.6f} |"
        )
    lines.extend(
        (
            "",
            "## Best model and clipping",
            "",
            f"Best primary-split result: **{best.model_evaluation.name} ({best.prediction_variant})** with MAE **{best_metrics.overall_mae_um:.6f} µm**, RMSE **{best_metrics.overall_rmse_um:.6f} µm**, and maximum absolute error **{best_metrics.max_absolute_error_um:.6f} µm**.",
            f"The first baseline MLP MAE was **{BASELINE_MLP_MAE_UM:.6f} µm**; v2 changes it by **{improvement:.6f} µm** ({100.0 * improvement / BASELINE_MLP_MAE_UM:.2f}% improvement when positive).",
            f"The first baseline maximum error was **{BASELINE_MLP_MAX_ABSOLUTE_ERROR_UM:.6f} µm**; the winning v2 result changes it by **{maximum_error_change:.6f} µm** (negative means v2 is worse).",
            "",
            "Clipping cannot increase coordinate-wise absolute error because every target lies within ±500 µm. "
            + (
                "It improves primary-split maximum error for: "
                + ", ".join(clipping_max_improved_models)
                + "."
                if clipping_max_improved_models
                else "It does not improve primary-split maximum error for any tested model."
            ),
            "",
            "## Recommendation",
            "",
            (
                "The winning v2 model should replace the first baseline because it improves both primary-split MAE and maximum error."
                if replacement
                else "Keep the tuned clipped MLP as a v2 candidate, but do not replace the previous baseline yet. Its MAE gain is small and its primary-split maximum error is worse; the requested physical closed-loop comparison has also deliberately not been run in this experiment."
            ),
            "",
            "This remains an underdetermined six-observation/eight-target inverse. Better held-out regression does not prove unique recovery of the physical electrode displacements. Run a separate closed-loop FEM validation before treating the replacement as physically superior.",
            "",
            "## Files",
            "",
            "- `metrics.csv`: primary raw/clipped model comparison.",
            "- `repeated_split_metrics.csv`: all five split/model/variant results.",
            "- `per_output_metrics.csv`: eight coordinate and four electrode-vector metrics.",
            "- `test_predictions.csv`: primary held-out predictions and errors.",
            "- `best_model.joblib`: winning estimator, including clipping when selected.",
            "- `plots/`: four requested diagnostic figures.",
            "",
        )
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the improved inverse-model training command line."""

    parser = argparse.ArgumentParser(
        prog="rf-trap-train-inverse-v2",
        description="Compare improved inverse models on the existing clean dataset.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("validation_results/generated_dataset/synthetic_clean.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation_results/inverse_model_v2"),
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def _print_progress(
    split_index: int,
    total_splits: int,
    model_name: str,
    status: str,
    runtime_seconds: float | None,
) -> None:
    if status == "started":
        print(
            f"split={split_index}/{total_splits} training={model_name}",
            flush=True,
        )
    else:
        print(
            f"split={split_index}/{total_splits} completed={model_name} "
            f"fit_seconds={runtime_seconds:.3f}",
            flush=True,
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Train the v2 comparison and write all requested artifacts."""

    arguments = build_parser().parse_args(argv)
    dataset = subset_inverse_dataset(
        load_v2_dataset(arguments.dataset),
        arguments.max_rows,
    )
    result = train_inverse_v2(
        dataset,
        test_size=arguments.test_size,
        random_state=arguments.random_state,
        repeat_count=arguments.repeats,
        smoke_test=arguments.smoke_test,
        progress_callback=_print_progress,
    )
    paths = write_v2_outputs(result, arguments.output_dir)
    best = result.best
    metrics = best.variant_evaluation.metrics
    raw_metrics = best.model_evaluation.raw.metrics
    clipped_metrics = best.model_evaluation.clipped.metrics
    print(f"best_model={best.model_evaluation.name}")
    print(f"best_prediction_variant={best.prediction_variant}")
    print(f"best_test_mae_um={metrics.overall_mae_um:.9g}")
    print(f"best_test_rmse_um={metrics.overall_rmse_um:.9g}")
    print(f"best_test_max_absolute_error_um={metrics.max_absolute_error_um:.9g}")
    print(
        "best_model_clipping_max_improvement_um="
        f"{raw_metrics.max_absolute_error_um - clipped_metrics.max_absolute_error_um:.9g}"
    )
    print(f"beats_baseline={metrics.overall_mae_um < BASELINE_MLP_MAE_UM}")
    print(f"metrics={paths.metrics_csv}")
    print(f"readme={paths.readme_markdown}")
    print(f"best_model_path={paths.best_model_joblib}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
