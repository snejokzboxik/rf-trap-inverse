"""ML-only learning curve for the existing merged inverse dataset."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

from .inverse_training import (
    InverseDataset,
    InverseMetrics,
    build_baseline_estimators,
    compute_inverse_metrics,
    load_inverse_dataset,
    split_inverse_dataset,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_DATASET = Path(
    "validation_results/generated_dataset_merged_29995/synthetic_clean_ml.csv"
)
DEFAULT_OUTPUT = Path("validation_results/learning_curve_merged_29995")
DEFAULT_DATASET_SIZES = (1000, 5000, 10000, 20000, 29995)


@dataclass(frozen=True)
class LearningCurvePoint:
    """One nested MLP fit evaluated on the common full held-out set."""

    requested_dataset_rows: int
    train_rows: int
    fixed_test_rows: int
    metrics: InverseMetrics
    fit_runtime_seconds: float


@dataclass(frozen=True)
class LearningCurveResult:
    """Deterministic nested learning-curve fits and their shared test design."""

    points: tuple[LearningCurvePoint, ...]
    total_dataset_rows: int
    full_train_pool_rows: int
    fixed_test_rows: int
    test_size: float
    random_state: int
    model_name: str


@dataclass(frozen=True)
class LearningCurveOutputPaths:
    """Files written by the learning-curve experiment."""

    metrics_csv: Path
    summary_json: Path
    readme_markdown: Path
    plot_directory: Path


ProgressCallback = Callable[[int, str, float | None], None]


def effective_training_rows(dataset_rows: int, test_size: float = 0.2) -> int:
    """Return sklearn-compatible training rows for a requested dataset size."""

    if dataset_rows < 2:
        raise ValueError("dataset_rows must be at least 2")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must lie strictly between zero and one")
    return dataset_rows - int(np.ceil(test_size * dataset_rows))


def validate_learning_curve_sizes(
    sizes: Sequence[int], total_rows: int
) -> tuple[int, ...]:
    """Validate a strictly increasing requested dataset-size schedule."""

    values = tuple(int(value) for value in sizes)
    if not values:
        raise ValueError("at least one learning-curve size is required")
    if any(value < 2 for value in values):
        raise ValueError("learning-curve sizes must be at least 2")
    if tuple(sorted(set(values))) != values:
        raise ValueError("learning-curve sizes must be unique and increasing")
    if values[-1] > total_rows:
        raise ValueError("learning-curve size exceeds the available dataset")
    return values


def run_learning_curve_experiment(
    dataset: InverseDataset,
    *,
    dataset_sizes: Sequence[int] = DEFAULT_DATASET_SIZES,
    test_size: float = 0.2,
    random_state: int = 42,
    smoke_test: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> LearningCurveResult:
    """Fit nested MLP training subsets against one fixed held-out test set.

    Requested sizes describe an equivalent 80/20 dataset size. The corresponding
    training row count is drawn from the full split's training pool, while every
    point is evaluated on the same full held-out set. This isolates the effect of
    adding training rows from test-set sampling noise.
    """

    sizes = validate_learning_curve_sizes(dataset_sizes, dataset.X_m.shape[0])
    split = split_inverse_dataset(
        dataset, test_size=test_size, random_state=random_state
    )
    order = np.random.default_rng(random_state).permutation(split.X_train_m.shape[0])
    points: list[LearningCurvePoint] = []
    for size in sizes:
        train_rows = effective_training_rows(size, test_size)
        if train_rows > split.X_train_m.shape[0]:
            raise ValueError("requested training rows exceed the full training pool")
        indices = order[:train_rows]
        estimator = build_baseline_estimators(
            random_state=random_state, smoke_test=smoke_test
        )["mlp"]
        if progress_callback is not None:
            progress_callback(size, "started", None)
        started = time.perf_counter()
        estimator.fit(split.X_train_m[indices], split.y_train_m[indices])
        runtime = time.perf_counter() - started
        prediction = np.asarray(estimator.predict(split.X_test_m), dtype=float)
        points.append(
            LearningCurvePoint(
                requested_dataset_rows=size,
                train_rows=train_rows,
                fixed_test_rows=split.X_test_m.shape[0],
                metrics=compute_inverse_metrics(split.y_test_m, prediction),
                fit_runtime_seconds=runtime,
            )
        )
        if progress_callback is not None:
            progress_callback(size, "completed", runtime)
    return LearningCurveResult(
        points=tuple(points),
        total_dataset_rows=dataset.X_m.shape[0],
        full_train_pool_rows=split.X_train_m.shape[0],
        fixed_test_rows=split.X_test_m.shape[0],
        test_size=test_size,
        random_state=random_state,
        model_name="mlp",
    )


def write_learning_curve_outputs(
    result: LearningCurveResult,
    output_directory: str | Path,
    *,
    dataset_source: str | Path,
) -> LearningCurveOutputPaths:
    """Write learning-curve metrics, plots, JSON summary, and README."""

    output = Path(output_directory)
    plots = output / "plots"
    output.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    paths = LearningCurveOutputPaths(
        metrics_csv=output / "learning_curve_metrics.csv",
        summary_json=output / "learning_curve_summary.json",
        readme_markdown=output / "README.md",
        plot_directory=plots,
    )
    rows = _metric_rows(result)
    with paths.metrics_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = _summary_payload(result, dataset_source)
    paths.summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot_mae_rmse(result, plots / "learning_curve_mae_rmse.png")
    _plot_maximum_error(result, plots / "learning_curve_maximum_error.png")
    paths.readme_markdown.write_text(
        _learning_curve_readme(result, summary), encoding="utf-8"
    )
    return paths


def run_learning_curve_from_csv(
    dataset_path: str | Path = DEFAULT_DATASET,
    output_directory: str | Path = DEFAULT_OUTPUT,
    *,
    dataset_sizes: Sequence[int] = DEFAULT_DATASET_SIZES,
    test_size: float = 0.2,
    random_state: int = 42,
    smoke_test: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> LearningCurveOutputPaths:
    """Load the existing merged CSV, fit the MLP curve, and write outputs."""

    dataset = load_inverse_dataset(dataset_path)
    result = run_learning_curve_experiment(
        dataset,
        dataset_sizes=dataset_sizes,
        test_size=test_size,
        random_state=random_state,
        smoke_test=smoke_test,
        progress_callback=progress_callback,
    )
    return write_learning_curve_outputs(
        result, output_directory, dataset_source=dataset_path
    )


def _metric_rows(result: LearningCurveResult) -> list[dict[str, object]]:
    rows = []
    first_mae = result.points[0].metrics.overall_mae_um
    previous_mae: float | None = None
    for point in result.points:
        metrics = point.metrics
        rows.append(
            {
                "model": result.model_name,
                "requested_dataset_rows": point.requested_dataset_rows,
                "train_rows": point.train_rows,
                "fixed_test_rows": point.fixed_test_rows,
                "overall_mae_um": metrics.overall_mae_um,
                "overall_rmse_um": metrics.overall_rmse_um,
                "max_absolute_error_um": metrics.max_absolute_error_um,
                "mean_electrode_vector_error_um": float(
                    np.mean(metrics.per_electrode_vector_error_mean_um)
                ),
                "max_electrode_vector_error_um": float(
                    np.max(metrics.per_electrode_vector_error_max_um)
                ),
                "mae_improvement_from_first_um": first_mae
                - metrics.overall_mae_um,
                "mae_improvement_from_previous_um": (
                    "" if previous_mae is None else previous_mae - metrics.overall_mae_um
                ),
                "fit_runtime_seconds": point.fit_runtime_seconds,
            }
        )
        previous_mae = metrics.overall_mae_um
    return rows


def _summary_payload(
    result: LearningCurveResult, dataset_source: str | Path
) -> dict[str, object]:
    first = result.points[0]
    last = result.points[-1]
    penultimate = result.points[-2] if len(result.points) > 1 else first
    return {
        "model": result.model_name,
        "dataset_source": str(dataset_source),
        "total_dataset_rows": result.total_dataset_rows,
        "full_train_pool_rows": result.full_train_pool_rows,
        "fixed_test_rows": result.fixed_test_rows,
        "test_size": result.test_size,
        "random_state": result.random_state,
        "requested_dataset_sizes": [
            point.requested_dataset_rows for point in result.points
        ],
        "first_mae_um": first.metrics.overall_mae_um,
        "final_mae_um": last.metrics.overall_mae_um,
        "absolute_mae_improvement_um": (
            first.metrics.overall_mae_um - last.metrics.overall_mae_um
        ),
        "relative_mae_improvement_percent": 100.0
        * (first.metrics.overall_mae_um - last.metrics.overall_mae_um)
        / first.metrics.overall_mae_um,
        "last_step_mae_improvement_um": (
            penultimate.metrics.overall_mae_um - last.metrics.overall_mae_um
        ),
        "points": _metric_rows(result),
        "method": (
            "nested training subsets from one deterministic full-data training pool; "
            "all points evaluated on the same full held-out test set"
        ),
        "model_artifacts_saved": False,
    }


def _plot_mae_rmse(result: LearningCurveResult, path: Path) -> None:
    x = [point.requested_dataset_rows for point in result.points]
    mae = [point.metrics.overall_mae_um for point in result.points]
    rmse = [point.metrics.overall_rmse_um for point in result.points]
    figure, axis = plt.subplots(figsize=(7.6, 4.9))
    axis.plot(x, mae, marker="o", linewidth=2.0, label="MAE")
    axis.plot(x, rmse, marker="s", linewidth=2.0, label="RMSE")
    axis.set_xlabel("Requested dataset size N")
    axis.set_ylabel("Held-out coordinate error (um)")
    axis.set_title("Merged-data MLP learning curve (fixed test set)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _plot_maximum_error(result: LearningCurveResult, path: Path) -> None:
    x = [point.requested_dataset_rows for point in result.points]
    maximum = [point.metrics.max_absolute_error_um for point in result.points]
    figure, axis = plt.subplots(figsize=(7.6, 4.9))
    axis.plot(x, maximum, marker="o", linewidth=2.0, color="#dc2626")
    axis.set_xlabel("Requested dataset size N")
    axis.set_ylabel("Maximum absolute coordinate error (um)")
    axis.set_title("Learning-curve worst coordinate error")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _learning_curve_readme(
    result: LearningCurveResult, summary: dict[str, object]
) -> str:
    lines = [
        "# Merged N=29995 MLP learning curve",
        "",
        "This ML-only experiment uses nested training subsets from one deterministic full-data training pool. Every point is evaluated on the same 5999-row held-out test set; no FEM solve, synthetic generation, or model artifact saving occurs.",
        "",
        "| Requested N | Training rows | Fixed test rows | MAE (um) | RMSE (um) | Max (um) | Fit time (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in result.points:
        lines.append(
            f"| {point.requested_dataset_rows} | {point.train_rows} | {point.fixed_test_rows} | "
            f"{point.metrics.overall_mae_um:.6f} | {point.metrics.overall_rmse_um:.6f} | "
            f"{point.metrics.max_absolute_error_um:.6f} | {point.fit_runtime_seconds:.3f} |"
        )
    lines.extend(
        (
            "",
            f"MAE changes from {summary['first_mae_um']:.6f} um at the first point to {summary['final_mae_um']:.6f} um at N={result.points[-1].requested_dataset_rows}, a {summary['relative_mae_improvement_percent']:.3f}% reduction.",
            "",
            "`learning_curve_metrics.csv` contains the exact values. Plots are under `plots/`. The curve measures synthetic held-out regression only and does not replace independent or closed-loop physical validation.",
            "",
        )
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the ML-only merged learning-curve command line."""

    parser = argparse.ArgumentParser(
        prog="rf-trap-learning-curve",
        description="Fit nested MLP subsets of the existing merged dataset.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=list(DEFAULT_DATASET_SIZES)
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use the lightweight test-only MLP settings.",
    )
    return parser


def _print_progress(size: int, status: str, runtime: float | None) -> None:
    if status == "started":
        print(f"training requested_N={size}...", flush=True)
    else:
        print(f"completed requested_N={size} in {runtime:.3f} s", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the learning curve and print its primary metrics path."""

    arguments = build_parser().parse_args(argv)
    paths = run_learning_curve_from_csv(
        arguments.dataset,
        arguments.output_dir,
        dataset_sizes=arguments.sizes,
        test_size=arguments.test_size,
        random_state=arguments.random_state,
        smoke_test=arguments.smoke_test,
        progress_callback=_print_progress,
    )
    summary = json.loads(paths.summary_json.read_text(encoding="utf-8"))
    print(f"first_mae_um={summary['first_mae_um']}")
    print(f"final_mae_um={summary['final_mae_um']}")
    print(f"metrics={paths.metrics_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
