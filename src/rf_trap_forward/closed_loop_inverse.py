"""Focused forward-FEM validation of a saved inverse displacement model."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .absolute_validation import wolfram_to_fem_absolute_displacements_m
from .config import ForwardModelConfig
from .dataset import sort_points_by_polar_angle
from .inverse_training import InverseDataset, load_inverse_dataset
from .minima_modes import RobustMinimaConfig
from .reference_validation import MinimumMatch, match_minima_by_distance
from .synthetic_dataset import (
    DEFAULT_AMBIGUOUS_MINIMUM_DISTANCE_M,
    PRACTICAL_CENTRAL_MESH_SIZE_M,
    SyntheticSolveResult,
    _default_fem_worker,
    minimum_pairwise_distance_m,
    practical_generator_forward_config,
)


DEFAULT_DATASET_PATH = Path(
    "validation_results/generated_dataset/synthetic_clean.csv"
)
DEFAULT_MODEL_PATH = Path(
    "validation_results/inverse_model_baseline/mlp.joblib"
)
DEFAULT_TEST_PREDICTIONS_PATH = Path(
    "validation_results/inverse_model_baseline/test_predictions.csv"
)
DEFAULT_OUTPUT_DIRECTORY = Path("validation_results/inverse_closed_loop")
DEFAULT_MODEL_NAME = "mlp"

CLOSED_LOOP_COLUMNS = (
    "sample_id",
    "selection_rank",
    "selection_source",
    "model",
    "status",
    "included_in_error_summary",
    "exactly_three_robust_minima",
    "true_min1_x_m",
    "true_min1_y_m",
    "true_min2_x_m",
    "true_min2_y_m",
    "true_min3_x_m",
    "true_min3_y_m",
    "predicted_w1_dx_m",
    "predicted_w1_dy_m",
    "predicted_w2_dx_m",
    "predicted_w2_dy_m",
    "predicted_w3_dx_m",
    "predicted_w3_dy_m",
    "predicted_w4_dx_m",
    "predicted_w4_dy_m",
    "fem_f1_dx_m",
    "fem_f1_dy_m",
    "fem_f2_dx_m",
    "fem_f2_dy_m",
    "fem_f3_dx_m",
    "fem_f3_dy_m",
    "fem_f4_dx_m",
    "fem_f4_dy_m",
    "recomputed_min1_x_m",
    "recomputed_min1_y_m",
    "recomputed_min2_x_m",
    "recomputed_min2_y_m",
    "recomputed_min3_x_m",
    "recomputed_min3_y_m",
    "match1_true_index",
    "match1_recomputed_index",
    "match1_error_um",
    "match2_true_index",
    "match2_recomputed_index",
    "match2_error_um",
    "match3_true_index",
    "match3_recomputed_index",
    "match3_error_um",
    "row_mean_error_um",
    "row_median_error_um",
    "row_max_error_um",
    "min_pairwise_distance_m",
    "accepted_candidate_count",
    "rejected_candidate_count",
    "total_candidate_count",
    "selected_interpolation_sensitive_count",
    "node_count",
    "triangle_count",
    "relative_free_residual",
    "runtime_seconds",
    "error_type",
    "error_message",
)


@dataclass(frozen=True)
class ClosedLoopSelection:
    """Deterministic sample IDs and the evidence used to choose them."""

    sample_ids: tuple[int, ...]
    source: str


@dataclass(frozen=True)
class PredictedDisplacements:
    """One model prediction in raw Wolfram and transformed FEM orders."""

    wolfram_displacements_m: NDArray[np.float64]
    fem_displacements_m: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate and copy both four-electrode displacement arrays."""

        wolfram = np.asarray(self.wolfram_displacements_m, dtype=float)
        fem = np.asarray(self.fem_displacements_m, dtype=float)
        if wolfram.shape != (4, 2) or fem.shape != (4, 2):
            raise ValueError("predicted displacements must have shape (4, 2)")
        if not np.all(np.isfinite(wolfram)) or not np.all(np.isfinite(fem)):
            raise ValueError("predicted displacements must be finite")
        object.__setattr__(self, "wolfram_displacements_m", wolfram.copy())
        object.__setattr__(self, "fem_displacements_m", fem.copy())


@dataclass(frozen=True)
class ClosedLoopRecord:
    """One selected sample, forward result, assignment, and audit diagnostics."""

    sample_id: int
    selection_rank: int
    selection_source: str
    model_name: str
    true_minima_m: NDArray[np.float64]
    prediction: PredictedDisplacements
    status: str
    included_in_error_summary: bool
    exactly_three_robust_minima: bool
    recomputed_minima_m: NDArray[np.float64]
    matches: tuple[MinimumMatch, ...]
    min_pairwise_distance_m: float
    solve: SyntheticSolveResult

    def __post_init__(self) -> None:
        """Validate immutable point arrays retained for reporting."""

        true = np.asarray(self.true_minima_m, dtype=float)
        recomputed = np.asarray(self.recomputed_minima_m, dtype=float)
        if true.shape != (3, 2) or not np.all(np.isfinite(true)):
            raise ValueError("true_minima_m must have finite shape (3, 2)")
        if recomputed.ndim != 2 or recomputed.shape[1] != 2:
            raise ValueError("recomputed_minima_m must have shape (n, 2)")
        if not np.all(np.isfinite(recomputed)):
            raise ValueError("recomputed minima must be finite")
        if self.included_in_error_summary and len(self.matches) != 3:
            raise ValueError("included rows must contain three Hungarian matches")
        object.__setattr__(self, "true_minima_m", true.copy())
        object.__setattr__(self, "recomputed_minima_m", recomputed.copy())

    def errors_um(self) -> NDArray[np.float64]:
        """Return available Hungarian assignment errors in micrometres."""

        return 1.0e6 * np.asarray(
            [match.distance_m for match in self.matches],
            dtype=float,
        )


@dataclass(frozen=True)
class ClosedLoopSummary:
    """Aggregate topology, failure, ambiguity, and accepted-error metrics."""

    requested_samples: int
    completed_forward_solves: int
    exactly_three_count: int
    solver_failure_count: int
    ambiguous_rejected_count: int
    ambiguous_branch_count: int
    robust_topology_rejected_count: int
    included_rows: int
    matched_minima: int
    mean_error_um: float | None
    median_error_um: float | None
    percentile_95_error_um: float | None
    maximum_error_um: float | None
    rows_with_rejected_candidates: int
    rejected_candidate_total: int
    selected_interpolation_sensitive_rows: int
    predicted_coordinates_outside_training_range: int
    maximum_absolute_predicted_displacement_um: float
    wall_runtime_seconds: float


@dataclass(frozen=True)
class ClosedLoopReport:
    """Complete small closed-loop experiment and configuration metadata."""

    records: tuple[ClosedLoopRecord, ...]
    selection: ClosedLoopSelection
    model_name: str
    model_path: Path
    dataset_path: Path
    forward_config: ForwardModelConfig
    ambiguity_threshold_m: float
    random_state: int
    wall_runtime_seconds: float

    def summary(self) -> ClosedLoopSummary:
        """Summarize only clean exactly-three rows in the error distribution."""

        errors = np.asarray(
            [
                error
                for record in self.records
                if record.included_in_error_summary
                for error in record.errors_um()
            ],
            dtype=float,
        )
        predicted = np.concatenate(
            [record.prediction.wolfram_displacements_m.ravel() for record in self.records]
        )
        return ClosedLoopSummary(
            requested_samples=len(self.records),
            completed_forward_solves=sum(not item.solve.error_type for item in self.records),
            exactly_three_count=sum(item.exactly_three_robust_minima for item in self.records),
            solver_failure_count=sum(item.status == "solver_failed" for item in self.records),
            ambiguous_rejected_count=sum(
                item.status in ("ambiguous_branch", "robust_topology_rejected")
                for item in self.records
            ),
            ambiguous_branch_count=sum(item.status == "ambiguous_branch" for item in self.records),
            robust_topology_rejected_count=sum(
                item.status == "robust_topology_rejected" for item in self.records
            ),
            included_rows=sum(item.included_in_error_summary for item in self.records),
            matched_minima=int(errors.size),
            mean_error_um=_optional_statistic(errors, np.mean),
            median_error_um=_optional_statistic(errors, np.median),
            percentile_95_error_um=(
                float(np.percentile(errors, 95.0)) if errors.size else None
            ),
            maximum_error_um=_optional_statistic(errors, np.max),
            rows_with_rejected_candidates=sum(
                item.solve.rejected_candidate_count > 0 for item in self.records
            ),
            rejected_candidate_total=sum(
                item.solve.rejected_candidate_count for item in self.records
            ),
            selected_interpolation_sensitive_rows=sum(
                item.solve.selected_interpolation_sensitive_count > 0
                for item in self.records
            ),
            predicted_coordinates_outside_training_range=int(
                np.count_nonzero(np.abs(predicted) > 500.0e-6)
            ),
            maximum_absolute_predicted_displacement_um=float(
                1.0e6 * np.max(np.abs(predicted))
            ),
            wall_runtime_seconds=self.wall_runtime_seconds,
        )


@dataclass(frozen=True)
class ClosedLoopOutputPaths:
    """Requested closed-loop artifact locations."""

    results_csv: Path
    summary_json: Path
    readme_markdown: Path


ClosedLoopWorker = Callable[
    [NDArray[np.float64], ForwardModelConfig, RobustMinimaConfig],
    SyntheticSolveResult,
]
ProgressCallback = Callable[[int, int, float], None]


def load_inverse_model(path: str | Path) -> object:
    """Load a trusted saved estimator and verify its prediction interface."""

    model = joblib.load(Path(path))
    if not callable(getattr(model, "predict", None)):
        raise TypeError("saved inverse model must expose predict(X)")
    return model


def predict_wolfram_displacements_m(
    model: object,
    minima_positions_m: ArrayLike,
) -> NDArray[np.float64]:
    """Predict one finite eight-coordinate raw Wolfram displacement vector."""

    minima = np.asarray(minima_positions_m, dtype=float)
    if minima.shape == (3, 2):
        model_input = minima.reshape(1, 6)
    elif minima.shape == (6,):
        model_input = minima.reshape(1, 6)
    else:
        raise ValueError("minima_positions_m must have shape (3, 2) or (6,)")
    if not np.all(np.isfinite(model_input)):
        raise ValueError("minima positions must be finite")
    prediction = np.asarray(model.predict(model_input), dtype=float)
    if prediction.shape == (8,):
        prediction = prediction.reshape(1, 8)
    if prediction.shape != (1, 8) or not np.all(np.isfinite(prediction)):
        raise ValueError("inverse model prediction must have finite shape (1, 8)")
    return prediction[0].copy()


def prepare_predicted_displacements(
    model: object,
    minima_positions_m: ArrayLike,
) -> PredictedDisplacements:
    """Predict raw W1--W4 displacements and apply ``-[W3,W1,W4,W2]``."""

    raw = predict_wolfram_displacements_m(model, minima_positions_m).reshape(4, 2)
    return PredictedDisplacements(
        wolfram_displacements_m=raw,
        fem_displacements_m=wolfram_to_fem_absolute_displacements_m(raw),
    )


def closed_loop_assignment_errors_um(
    true_minima_m: ArrayLike,
    recomputed_minima_m: ArrayLike,
) -> tuple[tuple[MinimumMatch, ...], NDArray[np.float64]]:
    """Hungarian-match two three-point sets and return distances in micrometres."""

    matches = match_minima_by_distance(true_minima_m, recomputed_minima_m)
    if len(matches) != 3:
        raise ValueError("closed-loop assignment requires exactly three minima")
    return matches, 1.0e6 * np.asarray(
        [match.distance_m for match in matches],
        dtype=float,
    )


def select_closed_loop_samples(
    dataset: InverseDataset,
    *,
    n: int = 20,
    test_predictions_path: str | Path | None = DEFAULT_TEST_PREDICTIONS_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    random_state: int = 42,
) -> ClosedLoopSelection:
    """Prefer saved held-out IDs, otherwise choose a fixed random subset."""

    if n <= 0 or n > dataset.sample_ids.size:
        raise ValueError("n must be positive and no larger than the dataset")
    available_ids = {int(value) for value in dataset.sample_ids}
    if test_predictions_path is not None and Path(test_predictions_path).is_file():
        identifiers = _test_prediction_ids(Path(test_predictions_path), model_name)
        missing = [value for value in identifiers if value not in available_ids]
        if missing:
            raise ValueError("saved test split contains sample IDs absent from the dataset")
        if len(identifiers) < n:
            raise ValueError("saved test split does not contain n distinct model rows")
        return ClosedLoopSelection(
            sample_ids=tuple(identifiers[:n]),
            source=f"saved test split IDs for {model_name}",
        )
    generator = np.random.default_rng(random_state)
    chosen = generator.choice(dataset.sample_ids, size=n, replace=False)
    return ClosedLoopSelection(
        sample_ids=tuple(int(value) for value in chosen),
        source=f"fixed random subset (random_state={random_state})",
    )


def run_closed_loop_validation(
    dataset: InverseDataset,
    model: object,
    selection: ClosedLoopSelection,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    batch_size: int = 4,
    ambiguity_threshold_m: float = DEFAULT_AMBIGUOUS_MINIMUM_DISTANCE_M,
    random_state: int = 42,
    worker: ClosedLoopWorker | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ClosedLoopReport:
    """Predict, transform, run robust FEM, and spatially match a small subset."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not np.isfinite(ambiguity_threshold_m) or ambiguity_threshold_m <= 0.0:
        raise ValueError("ambiguity_threshold_m must be finite and positive")
    if len(selection.sample_ids) == 0 or len(set(selection.sample_ids)) != len(
        selection.sample_ids
    ):
        raise ValueError("selection must contain distinct sample IDs")
    index_by_id = {
        int(sample_id): index for index, sample_id in enumerate(dataset.sample_ids)
    }
    if any(sample_id not in index_by_id for sample_id in selection.sample_ids):
        raise ValueError("selection contains a sample ID absent from the dataset")
    forward_config = practical_generator_forward_config("practical")
    robust_config = RobustMinimaConfig()
    selected_worker = worker or _default_fem_worker
    started = time.perf_counter()
    prepared = []
    for rank, sample_id in enumerate(selection.sample_ids, start=1):
        index = index_by_id[sample_id]
        true_minima = dataset.X_m[index].reshape(3, 2)
        prediction = prepare_predicted_displacements(model, true_minima)
        prepared.append((rank, sample_id, true_minima, prediction))
    records: dict[int, ClosedLoopRecord] = {}
    with ThreadPoolExecutor(max_workers=min(batch_size, len(prepared))) as executor:
        futures = {
            executor.submit(
                _safe_worker_call,
                selected_worker,
                prediction.fem_displacements_m,
                forward_config,
                robust_config,
            ): (rank, sample_id, true_minima, prediction)
            for rank, sample_id, true_minima, prediction in prepared
        }
        completed = 0
        for future in as_completed(futures):
            rank, sample_id, true_minima, prediction = futures[future]
            records[rank] = _record_from_solve(
                sample_id,
                rank,
                selection.source,
                model_name,
                true_minima,
                prediction,
                future.result(),
                ambiguity_threshold_m,
            )
            completed += 1
            if progress_callback is not None:
                progress_callback(
                    completed,
                    len(prepared),
                    time.perf_counter() - started,
                )
    return ClosedLoopReport(
        records=tuple(records[index] for index in range(1, len(prepared) + 1)),
        selection=selection,
        model_name=model_name,
        model_path=Path(model_path),
        dataset_path=Path(dataset_path),
        forward_config=forward_config,
        ambiguity_threshold_m=ambiguity_threshold_m,
        random_state=random_state,
        wall_runtime_seconds=time.perf_counter() - started,
    )


def write_closed_loop_outputs(
    report: ClosedLoopReport,
    output_directory: str | Path,
) -> ClosedLoopOutputPaths:
    """Write the requested closed-loop CSV, JSON summary, and Markdown report."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = ClosedLoopOutputPaths(
        results_csv=output / "closed_loop_results.csv",
        summary_json=output / "summary.json",
        readme_markdown=output / "README.md",
    )
    with paths.results_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(CLOSED_LOOP_COLUMNS))
        writer.writeheader()
        writer.writerows(_record_to_csv(item) for item in report.records)
    summary_record = _summary_record(report)
    paths.summary_json.write_text(
        json.dumps(summary_record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    paths.readme_markdown.write_text(
        _markdown_report(report, summary_record),
        encoding="utf-8",
    )
    return paths


def _test_prediction_ids(path: Path, model_name: str) -> list[int]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"model", "sample_id"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("test-predictions CSV lacks model/sample_id columns")
        identifiers = [
            int(row["sample_id"]) for row in reader if row["model"] == model_name
        ]
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("test-predictions CSV has missing or duplicate model IDs")
    return identifiers


def _safe_worker_call(
    worker: ClosedLoopWorker,
    fem_displacements_m: NDArray[np.float64],
    config: ForwardModelConfig,
    robust_config: RobustMinimaConfig,
) -> SyntheticSolveResult:
    try:
        result = worker(fem_displacements_m, config, robust_config)
    except Exception as error:
        return SyntheticSolveResult.failure(type(error).__name__, str(error))
    if not isinstance(result, SyntheticSolveResult):
        return SyntheticSolveResult.failure(
            "WorkerProtocolError",
            "closed-loop worker must return SyntheticSolveResult",
        )
    return result


def _record_from_solve(
    sample_id: int,
    rank: int,
    selection_source: str,
    model_name: str,
    true_minima_m: NDArray[np.float64],
    prediction: PredictedDisplacements,
    solve: SyntheticSolveResult,
    ambiguity_threshold_m: float,
) -> ClosedLoopRecord:
    recomputed = (
        sort_points_by_polar_angle(solve.minima_positions_m)
        if solve.minima_positions_m.size
        else np.empty((0, 2), dtype=float)
    )
    exactly_three = bool(
        not solve.error_type
        and solve.accepted_candidate_count == 3
        and recomputed.shape == (3, 2)
    )
    pairwise = (
        minimum_pairwise_distance_m(recomputed)
        if recomputed.shape[0] >= 2
        else float("nan")
    )
    matches: tuple[MinimumMatch, ...] = ()
    if recomputed.shape == (3, 2):
        matches, _ = closed_loop_assignment_errors_um(true_minima_m, recomputed)
    if solve.error_type:
        status = "solver_failed"
    elif not exactly_three:
        status = "robust_topology_rejected"
    elif pairwise < ambiguity_threshold_m:
        status = "ambiguous_branch"
    else:
        status = "ok"
    return ClosedLoopRecord(
        sample_id=sample_id,
        selection_rank=rank,
        selection_source=selection_source,
        model_name=model_name,
        true_minima_m=true_minima_m,
        prediction=prediction,
        status=status,
        included_in_error_summary=status == "ok",
        exactly_three_robust_minima=exactly_three,
        recomputed_minima_m=recomputed,
        matches=matches,
        min_pairwise_distance_m=pairwise,
        solve=solve,
    )


def _record_to_csv(record: ClosedLoopRecord) -> dict[str, object]:
    row: dict[str, object] = {
        "sample_id": record.sample_id,
        "selection_rank": record.selection_rank,
        "selection_source": record.selection_source,
        "model": record.model_name,
        "status": record.status,
        "included_in_error_summary": record.included_in_error_summary,
        "exactly_three_robust_minima": record.exactly_three_robust_minima,
        "min_pairwise_distance_m": _finite_or_blank(record.min_pairwise_distance_m),
        "accepted_candidate_count": record.solve.accepted_candidate_count,
        "rejected_candidate_count": record.solve.rejected_candidate_count,
        "total_candidate_count": record.solve.total_candidate_count,
        "selected_interpolation_sensitive_count": (
            record.solve.selected_interpolation_sensitive_count
        ),
        "node_count": record.solve.node_count,
        "triangle_count": record.solve.triangle_count,
        "relative_free_residual": _finite_or_blank(
            record.solve.relative_free_residual
        ),
        "runtime_seconds": record.solve.runtime_seconds,
        "error_type": record.solve.error_type,
        "error_message": record.solve.error_message,
    }
    _write_pairs(row, "true_min", record.true_minima_m, coordinate_names=("x", "y"))
    _write_pairs(
        row,
        "predicted_w",
        record.prediction.wolfram_displacements_m,
        coordinate_names=("dx", "dy"),
    )
    _write_pairs(
        row,
        "fem_f",
        record.prediction.fem_displacements_m,
        coordinate_names=("dx", "dy"),
    )
    _write_pairs(
        row,
        "recomputed_min",
        record.recomputed_minima_m,
        coordinate_names=("x", "y"),
        expected_pairs=3,
    )
    errors_um = record.errors_um()
    for index in range(3):
        if index < len(record.matches):
            match = record.matches[index]
            row[f"match{index + 1}_true_index"] = match.reference_index
            row[f"match{index + 1}_recomputed_index"] = match.computed_index
            row[f"match{index + 1}_error_um"] = errors_um[index]
        else:
            row[f"match{index + 1}_true_index"] = ""
            row[f"match{index + 1}_recomputed_index"] = ""
            row[f"match{index + 1}_error_um"] = ""
    row["row_mean_error_um"] = float(np.mean(errors_um)) if errors_um.size else ""
    row["row_median_error_um"] = (
        float(np.median(errors_um)) if errors_um.size else ""
    )
    row["row_max_error_um"] = float(np.max(errors_um)) if errors_um.size else ""
    return row


def _write_pairs(
    row: dict[str, object],
    prefix: str,
    values: NDArray[np.float64],
    *,
    coordinate_names: tuple[str, str],
    expected_pairs: int | None = None,
) -> None:
    pair_count = values.shape[0] if expected_pairs is None else expected_pairs
    for index in range(pair_count):
        for component, coordinate in enumerate(coordinate_names):
            key = f"{prefix}{index + 1}_{coordinate}_m"
            row[key] = float(values[index, component]) if index < values.shape[0] else ""


def _summary_record(report: ClosedLoopReport) -> dict[str, object]:
    summary = report.summary()
    return {
        "ambiguous_branch_count": summary.ambiguous_branch_count,
        "ambiguous_or_rejected_count": summary.ambiguous_rejected_count,
        "ambiguity_threshold_m": report.ambiguity_threshold_m,
        "central_mesh_size_m": PRACTICAL_CENTRAL_MESH_SIZE_M,
        "completed_forward_solves": summary.completed_forward_solves,
        "dataset_path": str(report.dataset_path),
        "error_summary_includes_status": "ok only",
        "exactly_three_count": summary.exactly_three_count,
        "included_rows": summary.included_rows,
        "matched_minima": summary.matched_minima,
        "maximum_absolute_predicted_displacement_um": (
            summary.maximum_absolute_predicted_displacement_um
        ),
        "maximum_error_um": summary.maximum_error_um,
        "mean_error_um": summary.mean_error_um,
        "median_error_um": summary.median_error_um,
        "mesh_mode": "practical 500 um central refinement",
        "model_name": report.model_name,
        "model_path": str(report.model_path),
        "percentile_95_error_um": summary.percentile_95_error_um,
        "predicted_coordinates_outside_training_range": (
            summary.predicted_coordinates_outside_training_range
        ),
        "random_state": report.random_state,
        "rejected_candidate_total": summary.rejected_candidate_total,
        "requested_samples": summary.requested_samples,
        "robust_topology_rejected_count": summary.robust_topology_rejected_count,
        "rows_with_rejected_candidates": summary.rows_with_rejected_candidates,
        "selected_interpolation_sensitive_rows": (
            summary.selected_interpolation_sensitive_rows
        ),
        "selected_sample_ids": list(report.selection.sample_ids),
        "selection_source": report.selection.source,
        "solver_failure_count": summary.solver_failure_count,
        "wall_runtime_seconds": summary.wall_runtime_seconds,
        "wolfram_to_fem_transform": "[-W3, -W1, -W4, -W2]",
    }


def _markdown_report(
    report: ClosedLoopReport,
    summary: dict[str, object],
) -> str:
    def metric(name: str) -> str:
        value = summary[name]
        return "not available" if value is None else f"{float(value):.6f} µm"

    clean_topology = (
        summary["exactly_three_count"] == summary["requested_samples"]
        and summary["included_rows"] == summary["requested_samples"]
        and summary["solver_failure_count"] == 0
        and summary["ambiguous_or_rejected_count"] == 0
    )
    submillimetre_tail = (
        summary["percentile_95_error_um"] is not None
        and float(summary["percentile_95_error_um"]) < 1000.0
    )
    if clean_topology and submillimetre_tail:
        assessment = (
            f"All {summary['requested_samples']} selected cases preserved the clean "
            "exactly-three topology, and the mean and 95th-percentile loop-closure "
            "errors are sub-millimetre. This makes the saved model physically useful "
            "for a coarse first demonstration."
        )
    else:
        assessment = (
            "This subset does not provide clean sub-millimetre topology-preserving "
            "closure for every case, so it is not yet reliable for a first demo."
        )

    lines = [
        "# Inverse-model closed-loop FEM validation",
        "",
        "This focused check uses the existing QA-passed synthetic CSV and saved MLP. "
        "It generates no data, fits no model, and runs no calibration or mesh sweep.",
        "",
        "## Method",
        "",
        f"- Selected samples: **{summary['requested_samples']}**, from "
        f"**{summary['selection_source']}**.",
        "- Input to inverse model: the original three polar-angle-sorted minima in metres.",
        "- Model output: eight raw displacement coordinates in Wolfram W1--W4 order.",
        "- No prediction clipping is applied.",
        "- FEM transform: `F1,F2,F3,F4 = -[W3,W1,W4,W2]`.",
        "- Forward model: real-scale, all-positive electrodes, fixed grounded 50 mm outer circle, robust minima mode, practical 500 µm central mesh.",
        "- Recomputed and original minima are compared by minimum-total-distance Hungarian assignment.",
        "- Aggregate errors include only `status=ok` rows: exactly three robust-accepted minima and pairwise separation at least 0.15 mm.",
        "",
        "## Results",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Mean closed-loop minimum error | {metric('mean_error_um')} |",
        f"| Median closed-loop minimum error | {metric('median_error_um')} |",
        f"| 95th-percentile error | {metric('percentile_95_error_um')} |",
        f"| Maximum error | {metric('maximum_error_um')} |",
        f"| Exactly-three robust topology | {summary['exactly_three_count']} / {summary['requested_samples']} |",
        f"| Included clean rows | {summary['included_rows']} / {summary['requested_samples']} |",
        f"| Solver failures | {summary['solver_failure_count']} |",
        f"| Ambiguous/rejected rows | {summary['ambiguous_or_rejected_count']} |",
        f"| Rows with robust-rejected extra candidates | {summary['rows_with_rejected_candidates']} |",
        f"| Selected interpolation-sensitive rows | {summary['selected_interpolation_sensitive_rows']} |",
        "",
        "## First-demo assessment",
        "",
        assessment + " It is not a precision inverse: "
        f"the worst matched minimum error is **{metric('maximum_error_um')}**, and the test "
        "does not establish unique recovery of the original eight displacement coordinates.",
        "",
        "## Prediction-range audit",
        "",
        f"The MLP produced **{summary['predicted_coordinates_outside_training_range']}** "
        "coordinates outside the generator's ±500 µm training range. The largest "
        f"absolute predicted coordinate was **{summary['maximum_absolute_predicted_displacement_um']:.6f} µm**. "
        "These values were not clipped before FEM evaluation.",
        "",
        "## Interpretation boundary",
        "",
        "This experiment evaluates physical loop closure in minimum-position space, not recovery of the unique original electrode displacements. Six minimum coordinates cannot generally identify eight independently sampled displacement coordinates uniquely. The CSV preserves every topology rejection, solver failure, assignment, and candidate diagnostic used by this summary.",
        "",
    ]
    return "\n".join(lines)


def _optional_statistic(
    values: NDArray[np.float64],
    function: Callable[[NDArray[np.float64]], np.floating],
) -> float | None:
    return float(function(values)) if values.size else None


def _finite_or_blank(value: float) -> float | str:
    return float(value) if np.isfinite(value) else ""


def build_parser() -> argparse.ArgumentParser:
    """Build the focused inverse closed-loop command line."""

    parser = argparse.ArgumentParser(
        prog="rf-trap-closed-loop-inverse",
        description="Run a small robust-FEM loop closure for a saved inverse model.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--test-predictions",
        type=Path,
        default=DEFAULT_TEST_PREDICTIONS_PATH,
        help="Saved prediction table used only to recover held-out sample IDs.",
    )
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    return parser


def _print_progress(completed: int, total: int, elapsed_seconds: float) -> None:
    print(
        f"completed={completed}/{total} elapsed_seconds={elapsed_seconds:.3f}",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Load the saved inverse, run N robust FEM cases, and write diagnostics."""

    arguments = build_parser().parse_args(argv)
    dataset = load_inverse_dataset(arguments.dataset)
    model = load_inverse_model(arguments.model)
    selection = select_closed_loop_samples(
        dataset,
        n=arguments.n,
        test_predictions_path=arguments.test_predictions,
        model_name=arguments.model_name,
        random_state=arguments.random_state,
    )
    report = run_closed_loop_validation(
        dataset,
        model,
        selection,
        model_name=arguments.model_name,
        model_path=arguments.model,
        dataset_path=arguments.dataset,
        batch_size=arguments.batch_size,
        random_state=arguments.random_state,
        progress_callback=_print_progress,
    )
    paths = write_closed_loop_outputs(report, arguments.output_dir)
    summary = report.summary()
    print(f"mean_error_um={summary.mean_error_um}")
    print(f"median_error_um={summary.median_error_um}")
    print(f"percentile_95_error_um={summary.percentile_95_error_um}")
    print(f"maximum_error_um={summary.maximum_error_um}")
    print(f"exactly_three_count={summary.exactly_three_count}")
    print(f"solver_failure_count={summary.solver_failure_count}")
    print(f"ambiguous_rejected_count={summary.ambiguous_rejected_count}")
    print(f"results={paths.results_csv}")
    print(f"summary={paths.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
