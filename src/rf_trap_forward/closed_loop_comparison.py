"""Like-for-like robust-FEM closed-loop comparison of the saved inverse models."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .closed_loop_inverse import (
    DEFAULT_DATASET_PATH,
    DEFAULT_TEST_PREDICTIONS_PATH,
    ClosedLoopReport,
    ClosedLoopSelection,
    ClosedLoopWorker,
    load_inverse_model,
    run_closed_loop_validation,
    select_closed_loop_samples,
)
from .inverse_model_artifacts import ClippedInverseModel
from .inverse_training import InverseDataset, load_inverse_dataset

DEFAULT_V1_MODEL_PATH = Path("validation_results/inverse_model_baseline/mlp.joblib")
DEFAULT_V2_MODEL_PATH = Path("validation_results/inverse_model_v2/best_model.joblib")
DEFAULT_OUTPUT_DIRECTORY = Path("validation_results/inverse_closed_loop_comparison")
TRAINING_BOUND_M = 500.0e-6

SUMMARY_COLUMNS = (
    "model", "model_path", "prediction_mode", "requested_samples",
    "completed_forward_solves", "exactly_three_count", "solver_failure_count",
    "ambiguous_rejected_count", "included_rows", "matched_minima",
    "mean_minima_error_um", "median_minima_error_um", "p95_minima_error_um",
    "max_minima_error_um", "raw_prediction_coordinates_outside_training_range",
    "reported_prediction_coordinates_outside_training_range", "wall_runtime_seconds",
)
PER_SAMPLE_COLUMNS = (
    "model", "sample_id", "selection_rank", "selection_source", "status",
    "included_in_error_summary", "exactly_three_robust_minima",
    "raw_prediction_coordinates_outside_training_range",
    "reported_prediction_coordinates_outside_training_range", "match1_error_um",
    "match2_error_um", "match3_error_um", "row_mean_error_um",
    "row_median_error_um", "row_max_error_um", "accepted_candidate_count",
    "rejected_candidate_count", "node_count", "triangle_count", "runtime_seconds",
    "error_type", "error_message",
)


@dataclass(frozen=True)
class ComparisonModelResult:
    """One saved estimator evaluated on the shared deterministic FEM subset."""

    name: str
    path: Path
    prediction_mode: str
    report: ClosedLoopReport
    raw_predictions_m: NDArray[np.float64]
    reported_predictions_m: NDArray[np.float64]

    def raw_outside_counts(self) -> NDArray[np.int64]:
        return np.count_nonzero(np.abs(self.raw_predictions_m) > TRAINING_BOUND_M, axis=1)

    def reported_outside_counts(self) -> NDArray[np.int64]:
        return np.count_nonzero(np.abs(self.reported_predictions_m) > TRAINING_BOUND_M, axis=1)


@dataclass(frozen=True)
class ClosedLoopComparison:
    """Both model reports with exactly one preserved sample selection."""

    selection: ClosedLoopSelection
    results: tuple[ComparisonModelResult, ComparisonModelResult]

    def v2_is_physically_better(self) -> bool:
        """Require clean equal topology and lower mean with no worse p95 or max."""

        v1, v2 = (item.report.summary() for item in self.results)
        if not (
            v1.exactly_three_count == v2.exactly_three_count == len(self.selection.sample_ids)
            and v1.solver_failure_count == v2.solver_failure_count == 0
            and v1.ambiguous_rejected_count == v2.ambiguous_rejected_count == 0
        ):
            return False
        values = (v1.mean_error_um, v1.percentile_95_error_um, v1.maximum_error_um,
                  v2.mean_error_um, v2.percentile_95_error_um, v2.maximum_error_um)
        return bool(all(value is not None for value in values)
                    and v2.mean_error_um < v1.mean_error_um
                    and v2.percentile_95_error_um <= v1.percentile_95_error_um
                    and v2.maximum_error_um <= v1.maximum_error_um)


def load_comparison_models(
    v1_path: str | Path = DEFAULT_V1_MODEL_PATH,
    v2_path: str | Path = DEFAULT_V2_MODEL_PATH,
) -> tuple[object, object]:
    """Load the retained v1 MLP and persisted clipped v2 tuned MLP."""

    return load_inverse_model(v1_path), load_inverse_model(v2_path)


def prediction_range_audit(
    model: object, X_m: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], str]:
    """Return pre-clip, actually reported predictions, and prediction mode."""

    if isinstance(model, ClippedInverseModel):
        raw, mode = np.asarray(model.estimator.predict(X_m), dtype=float), "clipped_to_+/-500_um"
    else:
        raw, mode = np.asarray(model.predict(X_m), dtype=float), "raw_unclipped"
    reported = np.asarray(model.predict(X_m), dtype=float)
    expected = (X_m.shape[0], 8)
    if raw.shape != expected or reported.shape != expected:
        raise ValueError("both saved inverse models must predict shape (N, 8)")
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(reported)):
        raise ValueError("saved inverse models must predict finite values")
    return raw, reported, mode


def run_closed_loop_comparison(
    dataset: InverseDataset,
    *,
    selection: ClosedLoopSelection,
    v1_model: object,
    v2_model: object,
    v1_path: str | Path = DEFAULT_V1_MODEL_PATH,
    v2_path: str | Path = DEFAULT_V2_MODEL_PATH,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    batch_size: int = 4,
    random_state: int = 42,
    worker: ClosedLoopWorker | None = None,
    progress_callback: Callable[[str, int, int, float], None] | None = None,
) -> ClosedLoopComparison:
    """Run both models through the same actual robust forward-FEM cases."""

    index_by_id = {int(sample_id): index for index, sample_id in enumerate(dataset.sample_ids)}
    X_selected = np.asarray([dataset.X_m[index_by_id[sample_id]] for sample_id in selection.sample_ids])
    specifications = (
        ("v1_baseline_mlp", Path(v1_path), v1_model),
        ("v2_tuned_clipped_mlp", Path(v2_path), v2_model),
    )
    results: list[ComparisonModelResult] = []
    for name, path, model in specifications:
        raw, reported, mode = prediction_range_audit(model, X_selected)
        def callback(done: int, total: int, elapsed: float, label: str = name) -> None:
            if progress_callback is not None:
                progress_callback(label, done, total, elapsed)
        report = run_closed_loop_validation(
            dataset, model, selection, model_name=name, model_path=path,
            dataset_path=dataset_path, batch_size=batch_size, random_state=random_state,
            worker=worker, progress_callback=callback if progress_callback else None,
        )
        results.append(ComparisonModelResult(name, path, mode, report, raw, reported))
    return ClosedLoopComparison(selection, (results[0], results[1]))


def write_comparison_outputs(
    comparison: ClosedLoopComparison, output_directory: str | Path
) -> tuple[Path, Path, Path]:
    """Write required CSVs, concise Markdown decision, and direct comparison plots."""

    output = Path(output_directory); plots = output / "plots"
    output.mkdir(parents=True, exist_ok=True); plots.mkdir(exist_ok=True)
    summary_path, rows_path, readme_path = output / "summary.csv", output / "per_sample_results.csv", output / "README.md"
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_COLUMNS); writer.writeheader()
        writer.writerows(_summary_row(result) for result in comparison.results)
    with rows_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PER_SAMPLE_COLUMNS); writer.writeheader()
        for result in comparison.results:
            writer.writerows(_per_sample_row(result, index) for index in range(len(result.report.records)))
    _write_plots(comparison, plots)
    readme_path.write_text(_markdown_report(comparison), encoding="utf-8")
    return summary_path, rows_path, readme_path


def _summary_row(result: ComparisonModelResult) -> dict[str, object]:
    summary = result.report.summary()
    return {
        "model": result.name, "model_path": str(result.path), "prediction_mode": result.prediction_mode,
        "requested_samples": summary.requested_samples, "completed_forward_solves": summary.completed_forward_solves,
        "exactly_three_count": summary.exactly_three_count, "solver_failure_count": summary.solver_failure_count,
        "ambiguous_rejected_count": summary.ambiguous_rejected_count, "included_rows": summary.included_rows,
        "matched_minima": summary.matched_minima, "mean_minima_error_um": summary.mean_error_um,
        "median_minima_error_um": summary.median_error_um, "p95_minima_error_um": summary.percentile_95_error_um,
        "max_minima_error_um": summary.maximum_error_um,
        "raw_prediction_coordinates_outside_training_range": int(np.sum(result.raw_outside_counts())),
        "reported_prediction_coordinates_outside_training_range": int(np.sum(result.reported_outside_counts())),
        "wall_runtime_seconds": summary.wall_runtime_seconds,
    }


def _per_sample_row(result: ComparisonModelResult, index: int) -> dict[str, object]:
    record = result.report.records[index]; errors = record.errors_um()
    row: dict[str, object] = {
        "model": result.name, "sample_id": record.sample_id, "selection_rank": record.selection_rank,
        "selection_source": record.selection_source, "status": record.status,
        "included_in_error_summary": record.included_in_error_summary,
        "exactly_three_robust_minima": record.exactly_three_robust_minima,
        "raw_prediction_coordinates_outside_training_range": int(result.raw_outside_counts()[index]),
        "reported_prediction_coordinates_outside_training_range": int(result.reported_outside_counts()[index]),
        "row_mean_error_um": float(np.mean(errors)) if errors.size else "",
        "row_median_error_um": float(np.median(errors)) if errors.size else "",
        "row_max_error_um": float(np.max(errors)) if errors.size else "",
        "accepted_candidate_count": record.solve.accepted_candidate_count,
        "rejected_candidate_count": record.solve.rejected_candidate_count,
        "node_count": record.solve.node_count, "triangle_count": record.solve.triangle_count,
        "runtime_seconds": record.solve.runtime_seconds, "error_type": record.solve.error_type,
        "error_message": record.solve.error_message,
    }
    for number in range(3):
        row[f"match{number + 1}_error_um"] = float(errors[number]) if number < errors.size else ""
    return row


def _write_plots(comparison: ClosedLoopComparison, plots: Path) -> None:
    labels = [result.name.replace("_", " ") for result in comparison.results]
    ranks = np.arange(1, len(comparison.selection.sample_ids) + 1)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for label, result in zip(labels, comparison.results, strict=True):
        values = [float(np.mean(record.errors_um())) if record.errors_um().size else np.nan for record in result.report.records]
        axis.plot(ranks, values, marker="o", label=label)
    axis.set(xlabel="Shared deterministic selection rank", ylabel="Per-sample mean closed-loop error (µm)")
    axis.grid(alpha=0.25); axis.legend(); figure.tight_layout()
    figure.savefig(plots / "per_sample_mean_error_comparison.png", dpi=160); plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    distributions = [np.concatenate([record.errors_um() for record in result.report.records if record.errors_um().size]) for result in comparison.results]
    axis.boxplot(distributions, tick_labels=labels, showfliers=True)
    axis.set_ylabel("Hungarian-matched minimum error (µm)"); axis.grid(axis="y", alpha=0.25); figure.tight_layout()
    figure.savefig(plots / "minimum_error_distribution_comparison.png", dpi=160); plt.close(figure)


def _markdown_report(comparison: ClosedLoopComparison) -> str:
    rows = [_summary_row(item) for item in comparison.results]
    preferred = "v2 tuned clipped MLP" if comparison.v2_is_physically_better() else "retain v1 baseline MLP"
    table = ["| Model | Mean (µm) | Median (µm) | p95 (µm) | Max (µm) | Exactly three | Failures | Ambiguous/rejected | Raw outside range |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        table.append(f"| {row['model']} | {float(row['mean_minima_error_um']):.6f} | {float(row['median_minima_error_um']):.6f} | {float(row['p95_minima_error_um']):.6f} | {float(row['max_minima_error_um']):.6f} | {row['exactly_three_count']} / {row['requested_samples']} | {row['solver_failure_count']} | {row['ambiguous_rejected_count']} | {row['raw_prediction_coordinates_outside_training_range']} |")
    return "\n".join([
        "# v1 vs v2 inverse-model closed-loop FEM comparison", "",
        "This fresh like-for-like check uses no new data, model fitting, calibration, or mesh sweep.", "",
        "## Shared protocol", "",
        f"- Sample IDs ({len(comparison.selection.sample_ids)}): `{', '.join(map(str, comparison.selection.sample_ids))}`.",
        f"- Selection: {comparison.selection.source}.",
        "- Each model predicts raw Wolfram W1--W4 displacements; FEM receives `-[W3, W1, W4, W2]`.",
        "- The forward check uses real-scale all-positive electrodes, a fixed grounded outer boundary, robust minima mode, and practical 500 µm central mesh.",
        "- Recomputed and original minimum sets are compared by Hungarian assignment.",
        "- v2 uses its persisted ±500 µm clipping; its raw pre-clipping excursions are reported separately.", "",
        "## Results", "", *table, "",
        "## Decision", "",
        f"**Recommendation: {preferred}.** Replacement requires clean equal topology, lower mean error, and no worse p95 or maximum error. This is deliberately stricter than held-out displacement MAE because this comparison measures physical loop closure.", "",
        "The CSV files preserve every selected case, status, candidate count, and per-minimum error. The plots compare the same ranked cases and all matched minima.", "",
    ])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rf-trap-compare-closed-loop-inverse", description="Compare saved v1/v2 inverse models with the same robust-FEM loop closure.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--v1-model", type=Path, default=DEFAULT_V1_MODEL_PATH)
    parser.add_argument("--v2-model", type=Path, default=DEFAULT_V2_MODEL_PATH)
    parser.add_argument("--test-predictions", type=Path, default=DEFAULT_TEST_PREDICTIONS_PATH)
    parser.add_argument("--n", type=int, default=20); parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4); parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    dataset = load_inverse_dataset(arguments.dataset)
    selection = select_closed_loop_samples(dataset, n=arguments.n, test_predictions_path=arguments.test_predictions, model_name="mlp", random_state=arguments.random_state)
    v1, v2 = load_comparison_models(arguments.v1_model, arguments.v2_model)
    comparison = run_closed_loop_comparison(dataset, selection=selection, v1_model=v1, v2_model=v2, v1_path=arguments.v1_model, v2_path=arguments.v2_model, dataset_path=arguments.dataset, batch_size=arguments.batch_size, random_state=arguments.random_state, progress_callback=lambda name, done, total, elapsed: print(f"model={name} completed={done}/{total} elapsed_seconds={elapsed:.3f}", flush=True))
    paths = write_comparison_outputs(comparison, arguments.output_dir)
    for result in comparison.results:
        row = _summary_row(result)
        print(f"{result.name}: mean_um={row['mean_minima_error_um']} median_um={row['median_minima_error_um']} p95_um={row['p95_minima_error_um']} max_um={row['max_minima_error_um']} exactly_three={row['exactly_three_count']} failures={row['solver_failure_count']} ambiguous={row['ambiguous_rejected_count']} raw_outside={row['raw_prediction_coordinates_outside_training_range']}")
    print(f"v2_physically_better={comparison.v2_is_physically_better()}")
    print(f"summary={paths[0]}\nper_sample={paths[1]}\nreport={paths[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
