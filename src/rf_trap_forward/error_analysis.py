"""Detailed ML and closed-loop error analysis from existing result CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from numpy.typing import NDArray
from scipy.stats import pearsonr, spearmanr

from .inverse_training import MICROMETRES_PER_METRE, TARGET_COLUMNS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_PREDICTIONS = Path(
    "validation_results/inverse_model_merged_29995/test_predictions.csv"
)
DEFAULT_CLOSED_LOOP = Path(
    "validation_results/closed_loop_inverse_merged_29995_n100/"
    "closed_loop_results.csv"
)
DEFAULT_DATASET = Path(
    "validation_results/generated_dataset_merged_29995/synthetic_clean_ml.csv"
)
DEFAULT_OUTPUT = Path("validation_results/error_analysis_merged_29995")
MINIMUM_COLUMNS = (
    "min1_x_m",
    "min1_y_m",
    "min2_x_m",
    "min2_y_m",
    "min3_x_m",
    "min3_y_m",
)


@dataclass(frozen=True)
class PredictionErrorData:
    """Held-out prediction errors for one inverse model."""

    model_name: str
    sample_ids: NDArray[np.int64]
    truth_m: NDArray[np.float64]
    predictions_m: NDArray[np.float64]

    @property
    def errors_um(self) -> NDArray[np.float64]:
        """Return signed coordinate errors in micrometres."""

        return MICROMETRES_PER_METRE * (self.predictions_m - self.truth_m)

    @property
    def vector_errors_um(self) -> NDArray[np.float64]:
        """Return four two-component displacement-vector errors per sample."""

        return np.linalg.norm(self.errors_um.reshape(-1, 4, 2), axis=2)


@dataclass(frozen=True)
class ClosedLoopAnalysis:
    """Per-case and matched-minimum closed-loop diagnostics."""

    case_rows: tuple[dict[str, object], ...]
    matched_errors_um: NDArray[np.float64]
    status_counts: Mapping[str, int]
    maximum_stored_error_discrepancy_um: float


@dataclass(frozen=True)
class ErrorAnalysisOutputPaths:
    """Files written by the merged error-analysis command."""

    per_coordinate_csv: Path
    per_electrode_csv: Path
    closed_loop_stats_csv: Path
    closed_loop_cases_csv: Path
    relationship_stats_csv: Path
    worst_cases_csv: Path
    summary_json: Path
    readme_markdown: Path
    plot_directory: Path


def load_prediction_error_data(
    path: str | Path,
    *,
    model_name: str = "mlp",
) -> PredictionErrorData:
    """Load one model's true and predicted Wolfram-order displacements."""

    rows = _read_csv(Path(path))
    required = {
        "model",
        "sample_id",
        *(f"true_{column}" for column in TARGET_COLUMNS),
        *(f"predicted_{column}" for column in TARGET_COLUMNS),
    }
    _require_columns(rows, required, Path(path))
    selected = [row for row in rows if row["model"] == model_name]
    if not selected:
        raise ValueError(f"no prediction rows found for model {model_name!r}")
    sample_ids = np.asarray([int(row["sample_id"]) for row in selected], dtype=np.int64)
    truth = np.asarray(
        [[float(row[f"true_{column}"]) for column in TARGET_COLUMNS] for row in selected],
        dtype=float,
    )
    predictions = np.asarray(
        [
            [float(row[f"predicted_{column}"]) for column in TARGET_COLUMNS]
            for row in selected
        ],
        dtype=float,
    )
    if len(np.unique(sample_ids)) != sample_ids.size:
        raise ValueError("prediction sample IDs must be unique within one model")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(predictions)):
        raise ValueError("prediction values must be finite")
    return PredictionErrorData(model_name, sample_ids, truth, predictions)


def build_closed_loop_analysis(
    closed_loop_path: str | Path,
    dataset_path: str | Path,
) -> ClosedLoopAnalysis:
    """Join closed-loop cases to true displacements and recompute all errors."""

    closed_path = Path(closed_loop_path)
    dataset_source = Path(dataset_path)
    closed_rows = _read_csv(closed_path)
    dataset_rows = _read_csv(dataset_source)
    _require_columns(
        closed_rows,
        {
            "sample_id",
            "status",
            "included_in_error_summary",
            "exactly_three_robust_minima",
            *(f"predicted_{column}" for column in TARGET_COLUMNS),
            *(f"true_{column}" for column in MINIMUM_COLUMNS),
            *(f"match{index}_error_um" for index in range(1, 4)),
            "row_mean_error_um",
            "row_median_error_um",
            "row_max_error_um",
            "min_pairwise_distance_m",
        },
        closed_path,
    )
    _require_columns(
        dataset_rows,
        {"sample_id", *TARGET_COLUMNS, *MINIMUM_COLUMNS, "min_pairwise_distance_m"},
        dataset_source,
    )
    dataset_by_id = {int(row["sample_id"]): row for row in dataset_rows}
    if len(dataset_by_id) != len(dataset_rows):
        raise ValueError("dataset sample IDs must be unique")
    status_counts = Counter(row["status"] for row in closed_rows)
    cases: list[dict[str, object]] = []
    matched_errors: list[float] = []
    maximum_discrepancy = 0.0
    for row in closed_rows:
        if row["included_in_error_summary"].lower() != "true":
            continue
        sample_id = int(row["sample_id"])
        if sample_id not in dataset_by_id:
            raise ValueError(f"closed-loop sample {sample_id} is missing from dataset")
        source = dataset_by_id[sample_id]
        errors = np.asarray(
            [float(row[f"match{index}_error_um"]) for index in range(1, 4)],
            dtype=float,
        )
        if not np.all(np.isfinite(errors)):
            raise ValueError(f"closed-loop sample {sample_id} has nonfinite errors")
        stored = np.asarray(
            [
                float(row["row_mean_error_um"]),
                float(row["row_median_error_um"]),
                float(row["row_max_error_um"]),
            ]
        )
        recomputed = np.asarray(
            [float(np.mean(errors)), float(np.median(errors)), float(np.max(errors))]
        )
        maximum_discrepancy = max(
            maximum_discrepancy,
            float(np.max(np.abs(stored - recomputed))),
        )
        true_displacements = np.asarray(
            [float(source[column]) for column in TARGET_COLUMNS], dtype=float
        )
        predicted_displacements = np.asarray(
            [float(row[f"predicted_{column}"]) for column in TARGET_COLUMNS],
            dtype=float,
        )
        minima = np.asarray(
            [float(row[f"true_{column}"]) for column in MINIMUM_COLUMNS], dtype=float
        )
        case: dict[str, object] = {
            "sample_id": sample_id,
            "status": row["status"],
            "exactly_three_robust_minima": (
                row["exactly_three_robust_minima"].lower() == "true"
            ),
            "match1_error_um": errors[0],
            "match2_error_um": errors[1],
            "match3_error_um": errors[2],
            "row_mean_error_um": recomputed[0],
            "row_median_error_um": recomputed[1],
            "row_max_error_um": recomputed[2],
            "true_displacement_norm_um": (
                MICROMETRES_PER_METRE * np.linalg.norm(true_displacements)
            ),
            "predicted_displacement_norm_um": (
                MICROMETRES_PER_METRE * np.linalg.norm(predicted_displacements)
            ),
            "displacement_prediction_error_norm_um": (
                MICROMETRES_PER_METRE
                * np.linalg.norm(predicted_displacements - true_displacements)
            ),
            "min_pairwise_distance_mm": (
                1.0e3 * float(row["min_pairwise_distance_m"])
            ),
            "max_absolute_minimum_coordinate_mm": 1.0e3 * np.max(np.abs(minima)),
        }
        cases.append(case)
        matched_errors.extend(errors.tolist())
    if not cases:
        raise ValueError("closed-loop file contains no included error-summary rows")
    return ClosedLoopAnalysis(
        tuple(cases),
        np.asarray(matched_errors, dtype=float),
        dict(status_counts),
        maximum_discrepancy,
    )


def write_error_analysis_outputs(
    predictions: PredictionErrorData,
    closed_loop: ClosedLoopAnalysis,
    output_directory: str | Path,
    *,
    prediction_source: str | Path,
    closed_loop_source: str | Path,
    dataset_source: str | Path,
) -> ErrorAnalysisOutputPaths:
    """Write analysis CSVs, plots, JSON summary, and a concise output README."""

    output = Path(output_directory)
    plots = output / "plots"
    output.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    paths = ErrorAnalysisOutputPaths(
        per_coordinate_csv=output / "per_coordinate_error_stats.csv",
        per_electrode_csv=output / "per_electrode_error_stats.csv",
        closed_loop_stats_csv=output / "closed_loop_error_stats.csv",
        closed_loop_cases_csv=output / "closed_loop_case_metrics.csv",
        relationship_stats_csv=output / "relationship_stats.csv",
        worst_cases_csv=output / "worst_10_closed_loop_cases.csv",
        summary_json=output / "analysis_summary.json",
        readme_markdown=output / "README.md",
        plot_directory=plots,
    )
    coordinate_rows = _coordinate_statistics(predictions)
    electrode_rows = _electrode_statistics(predictions)
    closed_rows = _closed_loop_statistics(closed_loop)
    relationship_rows = _relationship_statistics(closed_loop.case_rows)
    case_rows = list(closed_loop.case_rows)
    worst_rows = sorted(
        case_rows,
        key=lambda row: (float(row["row_max_error_um"]), float(row["row_mean_error_um"])),
        reverse=True,
    )[:10]
    worst_rows = [dict(rank=index, **row) for index, row in enumerate(worst_rows, 1)]
    _write_rows(paths.per_coordinate_csv, coordinate_rows)
    _write_rows(paths.per_electrode_csv, electrode_rows)
    _write_rows(paths.closed_loop_stats_csv, closed_rows)
    _write_rows(paths.closed_loop_cases_csv, case_rows)
    _write_rows(paths.relationship_stats_csv, relationship_rows)
    _write_rows(paths.worst_cases_csv, worst_rows)
    summary = _summary_payload(
        predictions,
        closed_loop,
        coordinate_rows,
        electrode_rows,
        relationship_rows,
        prediction_source=prediction_source,
        closed_loop_source=closed_loop_source,
        dataset_source=dataset_source,
    )
    paths.summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot_prediction_histogram(predictions, plots / "prediction_error_histogram.png")
    _plot_vector_histogram(predictions, plots / "vector_displacement_error_histogram.png")
    _plot_per_output_mae(coordinate_rows, plots / "per_output_mae.png")
    _plot_closed_loop_histogram(
        closed_loop, plots / "closed_loop_error_histogram.png"
    )
    _plot_relationship(
        case_rows,
        "true_displacement_norm_um",
        "True 8D displacement norm (um)",
        plots / "closed_loop_error_vs_displacement_norm.png",
    )
    _plot_relationship(
        case_rows,
        "min_pairwise_distance_mm",
        "Minimum pairwise minimum distance (mm)",
        plots / "closed_loop_error_vs_min_pairwise_distance.png",
    )
    _plot_relationship(
        case_rows,
        "max_absolute_minimum_coordinate_mm",
        "Maximum absolute input-minimum coordinate (mm)",
        plots / "closed_loop_error_vs_maximum_minimum_coordinate.png",
    )
    paths.readme_markdown.write_text(
        _analysis_readme(summary, paths), encoding="utf-8"
    )
    return paths


def run_error_analysis(
    prediction_path: str | Path = DEFAULT_PREDICTIONS,
    closed_loop_path: str | Path = DEFAULT_CLOSED_LOOP,
    dataset_path: str | Path = DEFAULT_DATASET,
    output_directory: str | Path = DEFAULT_OUTPUT,
    *,
    model_name: str = "mlp",
) -> ErrorAnalysisOutputPaths:
    """Run the complete read-only merged-result error analysis."""

    predictions = load_prediction_error_data(prediction_path, model_name=model_name)
    closed_loop = build_closed_loop_analysis(closed_loop_path, dataset_path)
    return write_error_analysis_outputs(
        predictions,
        closed_loop,
        output_directory,
        prediction_source=prediction_path,
        closed_loop_source=closed_loop_path,
        dataset_source=dataset_path,
    )


def _coordinate_statistics(data: PredictionErrorData) -> list[dict[str, object]]:
    errors = data.errors_um
    rows = []
    for index, coordinate in enumerate(TARGET_COLUMNS):
        values = errors[:, index]
        absolute = np.abs(values)
        rows.append(
            {
                "coordinate": coordinate.removesuffix("_m"),
                "electrode": f"W{index // 2 + 1}",
                "component": "dx" if index % 2 == 0 else "dy",
                "count": values.size,
                "signed_mean_error_um": float(np.mean(values)),
                "signed_std_error_um": float(np.std(values)),
                "mae_um": float(np.mean(absolute)),
                "median_absolute_error_um": float(np.median(absolute)),
                "p95_absolute_error_um": float(np.percentile(absolute, 95.0)),
                "rmse_um": float(np.sqrt(np.mean(np.square(values)))),
                "max_absolute_error_um": float(np.max(absolute)),
            }
        )
    return rows


def _electrode_statistics(data: PredictionErrorData) -> list[dict[str, object]]:
    errors = data.errors_um.reshape(-1, 4, 2)
    vectors = np.linalg.norm(errors, axis=2)
    rows = []
    for index in range(4):
        vector = vectors[:, index]
        components = errors[:, index, :]
        rows.append(
            {
                "electrode": f"W{index + 1}",
                "count": vector.size,
                "mean_vector_error_um": float(np.mean(vector)),
                "median_vector_error_um": float(np.median(vector)),
                "p95_vector_error_um": float(np.percentile(vector, 95.0)),
                "rmse_vector_error_um": float(np.sqrt(np.mean(np.square(vector)))),
                "max_vector_error_um": float(np.max(vector)),
                "mean_coordinate_absolute_error_um": float(
                    np.mean(np.abs(components))
                ),
                "dx_mae_um": float(np.mean(np.abs(components[:, 0]))),
                "dy_mae_um": float(np.mean(np.abs(components[:, 1]))),
                "dx_signed_bias_um": float(np.mean(components[:, 0])),
                "dy_signed_bias_um": float(np.mean(components[:, 1])),
            }
        )
    return rows


def _distribution_row(scope: str, values: NDArray[np.float64]) -> dict[str, object]:
    return {
        "scope": scope,
        "count": values.size,
        "minimum_um": float(np.min(values)),
        "mean_um": float(np.mean(values)),
        "median_um": float(np.median(values)),
        "standard_deviation_um": float(np.std(values)),
        "p95_um": float(np.percentile(values, 95.0)),
        "maximum_um": float(np.max(values)),
    }


def _closed_loop_statistics(analysis: ClosedLoopAnalysis) -> list[dict[str, object]]:
    cases = analysis.case_rows
    return [
        _distribution_row("matched_minimum_error", analysis.matched_errors_um),
        _distribution_row(
            "row_mean_error",
            np.asarray([float(row["row_mean_error_um"]) for row in cases]),
        ),
        _distribution_row(
            "row_max_error",
            np.asarray([float(row["row_max_error_um"]) for row in cases]),
        ),
    ]


def _relationship_statistics(
    cases: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    predictors = (
        ("true_displacement_norm_um", "um"),
        ("min_pairwise_distance_mm", "mm"),
        ("max_absolute_minimum_coordinate_mm", "mm"),
    )
    y = np.asarray([float(row["row_mean_error_um"]) for row in cases])
    rows = []
    for predictor, unit in predictors:
        x = np.asarray([float(row[predictor]) for row in cases])
        pearson = pearsonr(x, y)
        spearman = spearmanr(x, y)
        slope, intercept = np.polyfit(x, y, 1)
        rows.append(
            {
                "predictor": predictor,
                "predictor_unit": unit,
                "response": "row_mean_error_um",
                "count": x.size,
                "pearson_r": float(pearson.statistic),
                "pearson_p_value": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic),
                "spearman_p_value": float(spearman.pvalue),
                "linear_slope_response_um_per_predictor_unit": float(slope),
                "linear_intercept_um": float(intercept),
            }
        )
    return rows


def _summary_payload(
    predictions: PredictionErrorData,
    closed_loop: ClosedLoopAnalysis,
    coordinate_rows: Sequence[Mapping[str, object]],
    electrode_rows: Sequence[Mapping[str, object]],
    relationship_rows: Sequence[Mapping[str, object]],
    *,
    prediction_source: str | Path,
    closed_loop_source: str | Path,
    dataset_source: str | Path,
) -> dict[str, object]:
    errors = predictions.errors_um
    absolute = np.abs(errors)
    hardest_coordinate = max(coordinate_rows, key=lambda row: float(row["mae_um"]))
    hardest_electrode = max(
        electrode_rows, key=lambda row: float(row["mean_vector_error_um"])
    )
    return {
        "model": predictions.model_name,
        "prediction_rows": predictions.sample_ids.size,
        "prediction_coordinate_errors": absolute.size,
        "prediction_overall_mae_um": float(np.mean(absolute)),
        "prediction_overall_rmse_um": float(np.sqrt(np.mean(np.square(errors)))),
        "prediction_maximum_absolute_error_um": float(np.max(absolute)),
        "prediction_absolute_error_p95_um": float(np.percentile(absolute, 95.0)),
        "prediction_absolute_error_over_200_um_count": int(np.count_nonzero(absolute > 200.0)),
        "prediction_absolute_error_over_300_um_count": int(np.count_nonzero(absolute > 300.0)),
        "hardest_coordinate": dict(hardest_coordinate),
        "hardest_electrode": dict(hardest_electrode),
        "closed_loop_cases": len(closed_loop.case_rows),
        "closed_loop_matched_minima": closed_loop.matched_errors_um.size,
        "closed_loop_exactly_three_count": int(
            sum(bool(row["exactly_three_robust_minima"]) for row in closed_loop.case_rows)
        ),
        "closed_loop_mean_error_um": float(np.mean(closed_loop.matched_errors_um)),
        "closed_loop_median_error_um": float(np.median(closed_loop.matched_errors_um)),
        "closed_loop_p95_error_um": float(
            np.percentile(closed_loop.matched_errors_um, 95.0)
        ),
        "closed_loop_maximum_error_um": float(np.max(closed_loop.matched_errors_um)),
        "closed_loop_status_counts": dict(closed_loop.status_counts),
        "maximum_stored_error_discrepancy_um": (
            closed_loop.maximum_stored_error_discrepancy_um
        ),
        "relationships": [dict(row) for row in relationship_rows],
        "sources": {
            "test_predictions_csv": str(prediction_source),
            "closed_loop_results_csv": str(closed_loop_source),
            "merged_dataset_csv": str(dataset_source),
        },
        "units": {"displacement_error": "micrometres", "positions": "metres"},
    }


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty analysis table: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_prediction_histogram(data: PredictionErrorData, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    axis.hist(np.abs(data.errors_um).ravel(), bins=50, color="#2563eb", alpha=0.86)
    axis.set_xlabel("Absolute coordinate error (um)")
    axis.set_ylabel("Count")
    axis.set_title(f"{data.model_name.upper()} held-out coordinate errors")
    axis.grid(alpha=0.22)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _plot_vector_histogram(data: PredictionErrorData, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    colors = ("#2563eb", "#059669", "#d97706", "#dc2626")
    for index, color in enumerate(colors):
        axis.hist(
            data.vector_errors_um[:, index],
            bins=45,
            alpha=0.42,
            label=f"W{index + 1}",
            color=color,
        )
    axis.set_xlabel("Two-component displacement-vector error (um)")
    axis.set_ylabel("Count")
    axis.set_title("Held-out vector-error distributions by electrode")
    axis.legend()
    axis.grid(alpha=0.22)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _plot_per_output_mae(
    rows: Sequence[Mapping[str, object]], path: Path
) -> None:
    figure, axis = plt.subplots(figsize=(9.2, 4.9))
    labels = [str(row["coordinate"]).replace("_", " ") for row in rows]
    values = [float(row["mae_um"]) for row in rows]
    bars = axis.bar(labels, values, color="#2563eb")
    axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    axis.set_ylabel("MAE (um)")
    axis.set_title("Held-out MAE by Wolfram displacement coordinate")
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.22)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _plot_closed_loop_histogram(analysis: ClosedLoopAnalysis, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    axis.hist(analysis.matched_errors_um, bins=35, color="#7c3aed", alpha=0.86)
    axis.axvline(
        np.mean(analysis.matched_errors_um),
        color="#111827",
        linestyle="--",
        label=f"mean={np.mean(analysis.matched_errors_um):.1f} um",
    )
    axis.set_xlabel("Matched minimum-position error (um)")
    axis.set_ylabel("Count")
    axis.set_title("Closed-loop error distribution (300 matched minima)")
    axis.legend()
    axis.grid(alpha=0.22)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _plot_relationship(
    cases: Sequence[Mapping[str, object]],
    predictor: str,
    x_label: str,
    path: Path,
) -> None:
    x = np.asarray([float(row[predictor]) for row in cases])
    y = np.asarray([float(row["row_mean_error_um"]) for row in cases])
    slope, intercept = np.polyfit(x, y, 1)
    order = np.argsort(x)
    figure, axis = plt.subplots(figsize=(7.2, 4.9))
    axis.scatter(x, y, s=28, alpha=0.7, color="#2563eb")
    axis.plot(x[order], slope * x[order] + intercept, color="#dc2626", linewidth=1.7)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Closed-loop row-mean error (um)")
    axis.set_title("Closed-loop error relationship")
    axis.grid(alpha=0.22)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _analysis_readme(
    summary: Mapping[str, object], paths: ErrorAnalysisOutputPaths
) -> str:
    hardest_coordinate = summary["hardest_coordinate"]
    hardest_electrode = summary["hardest_electrode"]
    return "\n".join(
        (
            "# Merged N=29995 error-analysis outputs",
            "",
            "This directory is derived only from saved ML predictions, saved closed-loop results, and the existing merged dataset. It runs no FEM solve and changes no numerical result.",
            "",
            f"- MLP held-out rows: {summary['prediction_rows']}.",
            f"- Held-out coordinate MAE/RMSE/max: {summary['prediction_overall_mae_um']:.6f} / {summary['prediction_overall_rmse_um']:.6f} / {summary['prediction_maximum_absolute_error_um']:.6f} um.",
            f"- Hardest coordinate by MAE: {hardest_coordinate['coordinate']} ({hardest_coordinate['mae_um']:.6f} um).",
            f"- Hardest electrode by mean vector error: {hardest_electrode['electrode']} ({hardest_electrode['mean_vector_error_um']:.6f} um).",
            f"- Closed-loop matched-minimum mean/median/p95/max: {summary['closed_loop_mean_error_um']:.6f} / {summary['closed_loop_median_error_um']:.6f} / {summary['closed_loop_p95_error_um']:.6f} / {summary['closed_loop_maximum_error_um']:.6f} um.",
            "",
            "Tabular outputs:",
            "",
            f"- `{paths.per_coordinate_csv.name}`",
            f"- `{paths.per_electrode_csv.name}`",
            f"- `{paths.closed_loop_stats_csv.name}`",
            f"- `{paths.closed_loop_cases_csv.name}`",
            f"- `{paths.relationship_stats_csv.name}`",
            f"- `{paths.worst_cases_csv.name}`",
            f"- `{paths.summary_json.name}`",
            "",
            "Plots are under `plots/`. Correlations describe this deterministic N=100 closed-loop subset and do not establish physical causality.",
            "",
        )
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV header is missing: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def _require_columns(
    rows: Sequence[Mapping[str, str]], required: set[str], path: Path
) -> None:
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")


def build_parser() -> argparse.ArgumentParser:
    """Build the saved-result error-analysis command line."""

    parser = argparse.ArgumentParser(
        prog="rf-trap-error-analysis",
        description="Analyze existing merged inverse and closed-loop result CSVs.",
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--closed-loop", type=Path, default=DEFAULT_CLOSED_LOOP)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-name", default="mlp")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the analysis and print its primary result paths."""

    arguments = build_parser().parse_args(argv)
    paths = run_error_analysis(
        arguments.predictions,
        arguments.closed_loop,
        arguments.dataset,
        arguments.output_dir,
        model_name=arguments.model_name,
    )
    summary = json.loads(paths.summary_json.read_text(encoding="utf-8"))
    print(f"prediction_rows={summary['prediction_rows']}")
    print(f"prediction_mae_um={summary['prediction_overall_mae_um']}")
    print(f"closed_loop_mean_error_um={summary['closed_loop_mean_error_um']}")
    print(f"closed_loop_p95_error_um={summary['closed_loop_p95_error_um']}")
    print(f"summary={paths.summary_json}")
    print(f"worst_cases={paths.worst_cases_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
