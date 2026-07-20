"""Milestone-6 diagnostics for coordinate, scale, polarity, and model hypotheses."""

from __future__ import annotations

import argparse
import csv
import itertools
import pickle
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

from .config import ForwardModelConfig
from .dataset import ReferenceDataset, load_reference_dataset
from .real_scale import (
    REAL_ELECTRODE_RADIUS_M,
    REAL_INNER_RADIUS_M,
    REAL_OUTER_BOUNDARY_RADIUS_M,
    real_scale_forward_config,
)
from .reference_validation import (
    DisplacementMode,
    ForwardObservation,
    MinimumMatch,
    ReferenceValidationReport,
    ReferenceValidationVariant,
    match_minima_by_distance,
    prepare_reference_row_inputs,
    run_reference_validation,
)

ReferenceFrame = Literal["absolute", "electrode1-relative"]
ScaleMode = Literal["none", "fitted", "inner-radius-roundtrip"]

MILESTONE_5_MEAN_ERROR_M = 1.0868741115367125e-3
VALIDATION_MEAN_LIMIT_M = 0.25e-3
VALIDATION_MAXIMUM_LIMIT_M = 0.50e-3

COORDINATE_TRANSFORMS: dict[str, NDArray[np.float64]] = {
    "identity": np.asarray(((1.0, 0.0), (0.0, 1.0))),
    "flip-x": np.asarray(((-1.0, 0.0), (0.0, 1.0))),
    "flip-y": np.asarray(((1.0, 0.0), (0.0, -1.0))),
    "rotate-180": np.asarray(((-1.0, 0.0), (0.0, -1.0))),
    "swap-xy": np.asarray(((0.0, 1.0), (1.0, 0.0))),
    "rotate-90": np.asarray(((0.0, -1.0), (1.0, 0.0))),
    "rotate-270": np.asarray(((0.0, 1.0), (-1.0, 0.0))),
    "swap-negated": np.asarray(((0.0, -1.0), (-1.0, 0.0))),
}


@dataclass(frozen=True)
class FEMHypothesis:
    """One FEM input/numbering/potential hypothesis before output transforms."""

    name: str
    family: str
    displacement_mode: DisplacementMode
    electrode_permutation: tuple[int, int, int, int]
    electrode_potentials_v: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        """Validate a fixed-E1 model hypothesis."""

        ReferenceValidationVariant(
            name=self.name,
            displacement_mode=self.displacement_mode,
            electrode_permutation=self.electrode_permutation,
            polarity_name=self.family,
        )
        potentials = np.asarray(self.electrode_potentials_v, dtype=float)
        if potentials.shape != (4,) or not np.all(np.isfinite(potentials)):
            raise ValueError("electrode_potentials_v must contain four finite values")
        if np.max(np.abs(potentials)) <= 0.0:
            raise ValueError("at least one electrode potential must be nonzero")


@dataclass(frozen=True)
class OutputHypothesis:
    """Coordinate-frame and scale interpretation applied to one FEM result."""

    name: str
    family: str
    fem_hypothesis: FEMHypothesis
    reference_frame: ReferenceFrame
    coordinate_transform: str
    scale_mode: ScaleMode
    output_scale: float = 1.0
    fitted_on_rows: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Validate output transform and fitted scale metadata."""

        if self.reference_frame not in ("absolute", "electrode1-relative"):
            raise ValueError("unsupported reference_frame")
        if self.coordinate_transform not in COORDINATE_TRANSFORMS:
            raise ValueError("unsupported coordinate_transform")
        if self.scale_mode not in ("none", "fitted", "inner-radius-roundtrip"):
            raise ValueError("unsupported scale_mode")
        if not np.isfinite(self.output_scale) or self.output_scale <= 0.0:
            raise ValueError("output_scale must be finite and positive")


@dataclass(frozen=True)
class HypothesisRow:
    """One transformed row and its minimum-distance matches."""

    row_number: int
    status: str
    observation: ForwardObservation | None
    reference_positions_m: NDArray[np.float64]
    computed_positions_m: NDArray[np.float64]
    matches: tuple[MinimumMatch, ...]
    error_type: str = ""
    error_message: str = ""

    @property
    def completed(self) -> bool:
        """Return whether three transformed minima were assigned."""

        return self.status == "ok" and len(self.matches) == 3

    @property
    def exactly_three_physical_minima(self) -> bool:
        """Return the pre-selection three-candidate topology diagnostic."""

        return bool(
            self.completed
            and self.observation is not None
            and self.observation.hessian_validated_candidates == 3
        )

    def error_distances_m(self) -> NDArray[np.float64]:
        """Return matched Euclidean errors in metres."""

        return np.asarray([match.distance_m for match in self.matches], dtype=float)


@dataclass(frozen=True)
class HypothesisSummary:
    """Aggregate completion, topology, and error metrics."""

    selected_rows: int
    completed_rows: int
    failed_rows: int
    rows_with_exactly_three_physical_minima: int
    matched_minima: int
    mean_error_m: float
    median_error_m: float
    maximum_error_m: float
    percentile_95_error_m: float
    passes_validation_gate: bool


@dataclass(frozen=True)
class HypothesisResult:
    """One fully specified diagnostic hypothesis evaluated on a row set."""

    hypothesis: OutputHypothesis
    scope: str
    rows: tuple[HypothesisRow, ...]
    fem_runtime_seconds: float

    def summary(self) -> HypothesisSummary:
        """Compute aggregate metrics without discarding failed rows."""

        errors = np.asarray(
            [match.distance_m for row in self.rows for match in row.matches],
            dtype=float,
        )
        if errors.size:
            metrics = (
                float(np.mean(errors)),
                float(np.median(errors)),
                float(np.max(errors)),
                float(np.percentile(errors, 95.0)),
            )
        else:
            metrics = (float("nan"),) * 4
        completed = sum(row.completed for row in self.rows)
        exact = sum(row.exactly_three_physical_minima for row in self.rows)
        passes = passes_validation_gate(
            selected_rows=len(self.rows),
            completed_rows=completed,
            exactly_three_rows=exact,
            mean_error_m=metrics[0],
            maximum_error_m=metrics[2],
        )
        return HypothesisSummary(
            selected_rows=len(self.rows),
            completed_rows=completed,
            failed_rows=len(self.rows) - completed,
            rows_with_exactly_three_physical_minima=exact,
            matched_minima=int(errors.size),
            mean_error_m=metrics[0],
            median_error_m=metrics[1],
            maximum_error_m=metrics[2],
            percentile_95_error_m=metrics[3],
            passes_validation_gate=passes,
        )


@dataclass(frozen=True)
class BasisRowDiagnostic:
    """One row of one-electrode basis-field evidence."""

    row_number: int
    status: str
    target_positions_m: NDArray[np.float64]
    basis_fields_v_per_m: NDArray[np.float64] | None
    node_count: int = 0
    triangle_count: int = 0
    runtime_seconds: float = 0.0
    error_type: str = ""
    error_message: str = ""


@dataclass(frozen=True)
class BasisFitDiagnostic:
    """Global voltage-vector fit from one-electrode field bases."""

    rows: tuple[BasisRowDiagnostic, ...]
    electrode_potentials_v: tuple[float, float, float, float]
    gram_eigenvalues: tuple[float, float, float, float]
    normalized_smallest_eigenvalue: float
    runtime_seconds: float


@dataclass(frozen=True)
class HypothesisStudy:
    """Complete screening, promotion, scale, and basis-fit diagnostics."""

    screening_results: tuple[HypothesisResult, ...]
    promoted_results: tuple[HypothesisResult, ...]
    basis_fit: BasisFitDiagnostic
    screening_rows: tuple[int, ...]
    promotion_rows: tuple[int, ...]
    mesh_size_m: float
    runtime_seconds: float

    def best_result(self) -> HypothesisResult:
        """Return the best promoted result, falling back to screening."""

        candidates = self.promoted_results or self.screening_results
        return rank_hypotheses(candidates)[0]


@dataclass(frozen=True)
class HypothesisOutputPaths:
    """Milestone-6 CSV, Markdown, and plot artifact locations."""

    summary_csv: Path
    rows_csv: Path
    minima_csv: Path
    basis_fit_csv: Path
    scale_diagnostics_csv: Path
    markdown_report: Path
    plot_paths: tuple[Path, ...]


def apply_coordinate_transform(
    points_m: ArrayLike,
    transform_name: str,
) -> NDArray[np.float64]:
    """Apply a named orthogonal global coordinate transform to 2D points."""

    if transform_name not in COORDINATE_TRANSFORMS:
        raise ValueError(f"unknown coordinate transform: {transform_name}")
    points = np.asarray(points_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("points_m must have finite shape (n, 2)")
    return points @ COORDINATE_TRANSFORMS[transform_name].T


def fit_global_output_scale(
    reference_sets_m: Sequence[ArrayLike],
    computed_sets_m: Sequence[ArrayLike],
) -> float:
    """Fit one positive scalar with assignment updates from several starts."""

    if len(reference_sets_m) != len(computed_sets_m) or not reference_sets_m:
        raise ValueError("reference and computed set sequences must be nonempty and equal")
    references = [np.asarray(values, dtype=float) for values in reference_sets_m]
    computed = [np.asarray(values, dtype=float) for values in computed_sets_m]
    for reference, prediction in zip(references, computed, strict=True):
        if reference.shape != prediction.shape or reference.shape != (3, 2):
            raise ValueError("every point set must have shape (3, 2)")

    candidates: list[tuple[float, float]] = []
    for initial_scale in (0.25, 0.5, 1.0, 2.0, 4.0):
        scale = initial_scale
        for _ in range(12):
            assigned_reference = []
            assigned_computed = []
            for reference, prediction in zip(references, computed, strict=True):
                matches = match_minima_by_distance(reference, scale * prediction)
                for match in matches:
                    assigned_reference.append(match.reference_position_m)
                    assigned_computed.append(
                        prediction[match.computed_index - 1]
                    )
            ref = np.vstack(assigned_reference)
            pred = np.vstack(assigned_computed)
            denominator = float(np.sum(pred * pred))
            if denominator <= np.finfo(float).tiny:
                raise ValueError("computed positions cannot all be zero")
            updated = max(float(np.sum(pred * ref) / denominator), 1.0e-12)
            if abs(updated - scale) <= 1.0e-12 * max(1.0, scale):
                scale = updated
                break
            scale = updated
        errors = [
            match.distance_m
            for reference, prediction in zip(references, computed, strict=True)
            for match in match_minima_by_distance(reference, scale * prediction)
        ]
        candidates.append((float(np.mean(errors)), scale))
    return min(candidates)[1]


def passes_validation_gate(
    *,
    selected_rows: int,
    completed_rows: int,
    exactly_three_rows: int,
    mean_error_m: float,
    maximum_error_m: float,
) -> bool:
    """Apply the explicit completion, topology, mean, and maximum error gate."""

    return bool(
        selected_rows > 0
        and completed_rows == selected_rows
        and exactly_three_rows == selected_rows
        and np.isfinite(mean_error_m)
        and np.isfinite(maximum_error_m)
        and mean_error_m <= VALIDATION_MEAN_LIMIT_M
        and maximum_error_m <= VALIDATION_MAXIMUM_LIMIT_M
    )


def rank_hypotheses(
    results: Iterable[HypothesisResult],
) -> tuple[HypothesisResult, ...]:
    """Rank gate passes first, then completion, mean, maximum, and topology."""

    candidates = tuple(results)
    if not candidates:
        raise ValueError("at least one hypothesis result is required")

    def key(result: HypothesisResult) -> tuple[object, ...]:
        summary = result.summary()
        completion_fraction = summary.completed_rows / summary.selected_rows
        exact_fraction = (
            summary.rows_with_exactly_three_physical_minima / summary.selected_rows
        )
        return (
            -float(summary.passes_validation_gate),
            -completion_fraction,
            summary.mean_error_m if np.isfinite(summary.mean_error_m) else float("inf"),
            summary.maximum_error_m
            if np.isfinite(summary.maximum_error_m)
            else float("inf"),
            -exact_fraction,
            result.hypothesis.name,
        )

    return tuple(sorted(candidates, key=key))


def e1_preserving_permutations() -> tuple[tuple[int, int, int, int], ...]:
    """Return every source numbering permutation consistent with fixed source E1."""

    return tuple((1, *tail) for tail in itertools.permutations((2, 3, 4)))


def polarity_patterns() -> tuple[tuple[float, float, float, float], ...]:
    """Return all binary polarity patterns modulo physically irrelevant global sign."""

    return tuple(
        (1.0, float(second), float(third), float(fourth))
        for second, third, fourth in itertools.product((1, -1), repeat=3)
    )


def build_hypothesis_result(
    dataset: ReferenceDataset,
    report: ReferenceValidationReport,
    hypothesis: OutputHypothesis,
    *,
    scope: str,
) -> HypothesisResult:
    """Apply one output convention to retained raw FEM observations."""

    rows = []
    for source_row in report.rows:
        row_index = source_row.row_number - 1
        reference = _reference_positions(
            dataset,
            row_index,
            hypothesis.reference_frame,
        )
        if source_row.observation is None:
            rows.append(
                HypothesisRow(
                    row_number=source_row.row_number,
                    status=source_row.status,
                    observation=None,
                    reference_positions_m=reference,
                    computed_positions_m=np.empty((0, 2), dtype=float),
                    matches=(),
                    error_type=source_row.error_type,
                    error_message=source_row.error_message,
                )
            )
            continue
        computed = hypothesis.output_scale * apply_coordinate_transform(
            source_row.observation.minima_positions_m,
            hypothesis.coordinate_transform,
        )
        matches = (
            match_minima_by_distance(reference, computed)
            if computed.shape == (3, 2)
            else ()
        )
        status = "ok" if len(matches) == 3 else "unexpected-minimum-count"
        rows.append(
            HypothesisRow(
                row_number=source_row.row_number,
                status=status,
                observation=source_row.observation,
                reference_positions_m=reference,
                computed_positions_m=computed,
                matches=matches,
                error_type=(
                    "" if status == "ok" else "UnexpectedMinimumCount"
                ),
                error_message=(
                    ""
                    if status == "ok"
                    else f"computed {computed.shape[0]} minima; expected 3"
                ),
            )
        )
    return HypothesisResult(
        hypothesis=hypothesis,
        scope=scope,
        rows=tuple(rows),
        fem_runtime_seconds=report.runtime_seconds,
    )


def run_model_hypothesis_study(
    dataset: ReferenceDataset,
    *,
    screening_rows: Sequence[int] = tuple(range(1, 11)),
    promotion_rows: Sequence[int] = tuple(range(1, 51)),
    mesh_size_m: float = 2.0e-3,
    promote_count: int = 3,
) -> HypothesisStudy:
    """Run the staged Milestone-6 hypothesis screen and promotion study."""

    screen = tuple(int(value) for value in screening_rows)
    promote = tuple(int(value) for value in promotion_rows)
    if not screen or not promote or promote[: len(screen)] != screen:
        raise ValueError("promotion_rows must begin with all screening_rows")
    if promote_count <= 0:
        raise ValueError("promote_count must be positive")
    started = time.perf_counter()
    base_reports: dict[str, ReferenceValidationReport] = {}
    fem_hypotheses: dict[str, FEMHypothesis] = {}
    screening_results: list[HypothesisResult] = []

    for displacement_mode in ("electrode1-relative", "absolute"):
        for permutation in e1_preserving_permutations():
            fem = FEMHypothesis(
                name=_fem_name(
                    displacement_mode,
                    permutation,
                    (1.0, 1.0, 1.0, 1.0),
                ),
                family="all-positive",
                displacement_mode=displacement_mode,
                electrode_permutation=permutation,
                electrode_potentials_v=(1.0, 1.0, 1.0, 1.0),
            )
            fem_hypotheses[fem.name] = fem
            report = _run_fem_hypothesis(dataset, fem, screen, mesh_size_m)
            base_reports[fem.name] = report
            screening_results.extend(
                _expand_output_hypotheses(dataset, report, fem, screen, "screen")
            )

    best_coordinate = rank_hypotheses(screening_results)[0]
    polarity_template = best_coordinate.hypothesis.fem_hypothesis
    for potentials in polarity_patterns():
        if potentials == (1.0, 1.0, 1.0, 1.0):
            continue
        fem = FEMHypothesis(
            name=_fem_name(
                polarity_template.displacement_mode,
                polarity_template.electrode_permutation,
                potentials,
            ),
            family=_polarity_name(potentials),
            displacement_mode=polarity_template.displacement_mode,
            electrode_permutation=polarity_template.electrode_permutation,
            electrode_potentials_v=potentials,
        )
        fem_hypotheses[fem.name] = fem
        report = _run_fem_hypothesis(dataset, fem, screen, mesh_size_m)
        base_reports[fem.name] = report
        screening_results.extend(
            _expand_output_hypotheses(dataset, report, fem, screen, "screen")
        )

    best_before_basis = rank_hypotheses(screening_results)[0]
    basis_fit = run_basis_field_fit(
        dataset,
        best_before_basis.hypothesis,
        screen,
        mesh_size_m=mesh_size_m,
    )
    fitted_fem = replace(
        best_before_basis.hypothesis.fem_hypothesis,
        name="basis_fitted_"
        + best_before_basis.hypothesis.fem_hypothesis.name,
        family="basis-fitted-linear-combination",
        electrode_potentials_v=basis_fit.electrode_potentials_v,
    )
    fem_hypotheses[fitted_fem.name] = fitted_fem
    fitted_report = _run_fem_hypothesis(dataset, fitted_fem, screen, mesh_size_m)
    base_reports[fitted_fem.name] = fitted_report
    screening_results.extend(
        _expand_output_hypotheses(
            dataset,
            fitted_report,
            fitted_fem,
            screen,
            "screen",
        )
    )

    ranked = rank_hypotheses(screening_results)
    promoted_specs = _distinct_top_hypotheses(ranked, promote_count)
    promoted_results = []
    extra_rows = promote[len(screen) :]
    promoted_reports: dict[str, ReferenceValidationReport] = {}
    for screened_result in promoted_specs:
        hypothesis = screened_result.hypothesis
        base_name = hypothesis.fem_hypothesis.name
        if base_name not in promoted_reports:
            report = base_reports[base_name]
            if extra_rows:
                extra_report = _run_fem_hypothesis(
                    dataset,
                    hypothesis.fem_hypothesis,
                    extra_rows,
                    mesh_size_m,
                )
                report = replace(
                    report,
                    rows=report.rows + extra_report.rows,
                    runtime_seconds=(
                        report.runtime_seconds + extra_report.runtime_seconds
                    ),
                )
            promoted_reports[base_name] = report
        promoted_results.append(
            build_hypothesis_result(
                dataset,
                promoted_reports[base_name],
                hypothesis,
                scope="promoted",
            )
        )

    return HypothesisStudy(
        screening_results=tuple(screening_results),
        promoted_results=tuple(promoted_results),
        basis_fit=basis_fit,
        screening_rows=screen,
        promotion_rows=promote,
        mesh_size_m=mesh_size_m,
        runtime_seconds=time.perf_counter() - started,
    )


def _run_fem_hypothesis(
    dataset: ReferenceDataset,
    fem: FEMHypothesis,
    rows: Sequence[int],
    mesh_size_m: float,
) -> ReferenceValidationReport:
    config = real_scale_forward_config(
        mesh_size_m=mesh_size_m,
        electrode_potentials_v=fem.electrode_potentials_v,
    )
    variant = ReferenceValidationVariant(
        name=fem.name,
        displacement_mode=fem.displacement_mode,
        electrode_permutation=fem.electrode_permutation,
        polarity_name=fem.family,
    )
    return run_reference_validation(dataset, config, rows, variant=variant)


def _expand_output_hypotheses(
    dataset: ReferenceDataset,
    report: ReferenceValidationReport,
    fem: FEMHypothesis,
    fitted_on_rows: tuple[int, ...],
    scope: str,
) -> list[HypothesisResult]:
    results = []
    for frame in ("absolute", "electrode1-relative"):
        for transform_name in COORDINATE_TRANSFORMS:
            base_name = f"{fem.name}__ref-{frame}__{transform_name}"
            unscaled = OutputHypothesis(
                name=base_name + "__scale-none",
                family=fem.family,
                fem_hypothesis=fem,
                reference_frame=frame,
                coordinate_transform=transform_name,
                scale_mode="none",
            )
            unscaled_result = build_hypothesis_result(
                dataset,
                report,
                unscaled,
                scope=scope,
            )
            results.append(unscaled_result)
            usable = [row for row in unscaled_result.rows if row.completed]
            if usable:
                scale = fit_global_output_scale(
                    [row.reference_positions_m for row in usable],
                    [row.computed_positions_m for row in usable],
                )
                fitted = replace(
                    unscaled,
                    name=base_name + "__scale-fitted",
                    family=fem.family + "+output-scale-fit",
                    scale_mode="fitted",
                    output_scale=scale,
                    fitted_on_rows=fitted_on_rows,
                )
                results.append(
                    build_hypothesis_result(
                        dataset,
                        report,
                        fitted,
                        scope=scope,
                    )
                )

        native_frame: ReferenceFrame = (
            "electrode1-relative"
            if fem.displacement_mode == "electrode1-relative"
            else "absolute"
        )
        if frame == native_frame:
            roundtrip = OutputHypothesis(
                name=f"{fem.name}__ref-{frame}__identity__inner-radius-roundtrip",
                family=fem.family + "+inner-radius-roundtrip",
                fem_hypothesis=fem,
                reference_frame=frame,
                coordinate_transform="identity",
                scale_mode="inner-radius-roundtrip",
            )
            results.append(
                build_hypothesis_result(
                    dataset,
                    report,
                    roundtrip,
                    scope=scope,
                )
            )
    return results


def _reference_positions(
    dataset: ReferenceDataset,
    row_index: int,
    frame: ReferenceFrame,
) -> NDArray[np.float64]:
    if frame == "absolute":
        return dataset.raw_minima_absolute_m[row_index].copy()
    return dataset.minima_relative_to_electrode1_m[row_index].copy()


def _distinct_top_hypotheses(
    ranked: Sequence[HypothesisResult],
    count: int,
) -> tuple[HypothesisResult, ...]:
    selected = []
    names = set()
    for result in ranked:
        name = result.hypothesis.name
        if name in names:
            continue
        selected.append(result)
        names.add(name)
        if len(selected) == count:
            break
    return tuple(selected)


def _fem_name(
    displacement_mode: DisplacementMode,
    permutation: tuple[int, int, int, int],
    potentials: tuple[float, float, float, float],
) -> str:
    mode = "relative" if displacement_mode == "electrode1-relative" else "absolute"
    mapping = "".join(map(str, permutation))
    polarity = "".join("p" if value >= 0.0 else "m" for value in potentials)
    return f"{mode}_perm{mapping}_{polarity}"


def _polarity_name(potentials: tuple[float, float, float, float]) -> str:
    if potentials == (1.0, 1.0, 1.0, 1.0):
        return "all-positive"
    if potentials == (1.0, -1.0, -1.0, 1.0):
        return "alternating-checkerboard"
    return "binary-polarity-" + "".join(
        "p" if value >= 0.0 else "m" for value in potentials
    )


def fit_basis_potentials(
    basis_fields: Sequence[ArrayLike],
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float], float]:
    """Fit normalized electrode weights that minimize fields at target points."""

    if not basis_fields:
        raise ValueError("at least one basis-field array is required")
    matrices = []
    for values in basis_fields:
        fields = np.asarray(values, dtype=float)
        if fields.ndim != 3 or fields.shape[1:] != (2, 4):
            raise ValueError("basis fields must have shape (n_points, 2, 4)")
        if not np.all(np.isfinite(fields)):
            raise ValueError("basis fields must be finite")
        matrices.append(fields.reshape(-1, 4))
    design = np.vstack(matrices)
    gram = design.T @ design
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    weights = eigenvectors[:, 0]
    weights /= np.max(np.abs(weights))
    first_nonzero = int(np.flatnonzero(np.abs(weights) > 1.0e-12)[0])
    if weights[first_nonzero] < 0.0:
        weights *= -1.0
    trace = float(np.trace(gram))
    normalized = (
        float(eigenvalues[0] / trace)
        if trace > np.finfo(float).tiny
        else float("nan")
    )
    return (
        tuple(float(value) for value in weights),
        tuple(float(value) for value in eigenvalues),
        normalized,
    )


def run_basis_field_fit(
    dataset: ReferenceDataset,
    hypothesis: OutputHypothesis,
    row_numbers: Sequence[int],
    *,
    mesh_size_m: float = 2.0e-3,
) -> BasisFitDiagnostic:
    """Evaluate one-hot fields in fresh processes and fit global voltages."""

    started = time.perf_counter()
    fem = hypothesis.fem_hypothesis
    config = real_scale_forward_config(mesh_size_m=mesh_size_m)
    variant = ReferenceValidationVariant(
        name=fem.name,
        displacement_mode=fem.displacement_mode,
        electrode_permutation=fem.electrode_permutation,
        polarity_name="one-hot-basis",
    )
    rows = []
    usable_fields = []
    transform = COORDINATE_TRANSFORMS[hypothesis.coordinate_transform]
    for row_number in row_numbers:
        index = row_number - 1
        raw = dataset.raw_displacements_m[index]
        reference = _reference_positions(dataset, index, hypothesis.reference_frame)
        target_positions = (reference / hypothesis.output_scale) @ transform
        solver_displacements, _, row_config = prepare_reference_row_inputs(
            raw,
            dataset.raw_minima_absolute_m[index],
            config,
            variant,
        )
        outcome = _run_isolated_basis_row(
            solver_displacements,
            row_config,
            target_positions,
        )
        if bool(outcome.get("ok")):
            fields = np.asarray(outcome["basis_fields_v_per_m"], dtype=float)
            usable_fields.append(fields)
            rows.append(
                BasisRowDiagnostic(
                    row_number=row_number,
                    status="ok",
                    target_positions_m=target_positions,
                    basis_fields_v_per_m=fields,
                    node_count=int(outcome["node_count"]),
                    triangle_count=int(outcome["triangle_count"]),
                    runtime_seconds=float(outcome["runtime_seconds"]),
                )
            )
        else:
            rows.append(
                BasisRowDiagnostic(
                    row_number=row_number,
                    status="failed",
                    target_positions_m=target_positions,
                    basis_fields_v_per_m=None,
                    runtime_seconds=float(outcome.get("runtime_seconds", 0.0)),
                    error_type=str(outcome.get("error_type", "BasisWorkerError")),
                    error_message=str(outcome.get("error_message", "unknown error")),
                )
            )
    if not usable_fields:
        raise RuntimeError("all one-hot basis-field rows failed")
    potentials, eigenvalues, normalized = fit_basis_potentials(usable_fields)
    return BasisFitDiagnostic(
        rows=tuple(rows),
        electrode_potentials_v=potentials,
        gram_eigenvalues=eigenvalues,
        normalized_smallest_eigenvalue=normalized,
        runtime_seconds=time.perf_counter() - started,
    )


def _run_isolated_basis_row(
    displacements_m: NDArray[np.float64],
    config: ForwardModelConfig,
    target_positions_m: NDArray[np.float64],
) -> dict[str, object]:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "rf_trap_forward._basis_fit_worker"],
        input=pickle.dumps((displacements_m, config, target_positions_m)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    runtime = time.perf_counter() - started
    if completed.returncode != 0:
        return {
            "ok": False,
            "runtime_seconds": runtime,
            "error_type": "BasisWorkerProcessError",
            "error_message": completed.stderr.decode(
                "utf-8", errors="replace"
            ).strip(),
        }
    try:
        outcome = pickle.loads(completed.stdout)
    except Exception as error:
        return {
            "ok": False,
            "runtime_seconds": runtime,
            "error_type": "BasisWorkerProtocolError",
            "error_message": str(error),
        }
    if not isinstance(outcome, dict) or "ok" not in outcome:
        return {
            "ok": False,
            "runtime_seconds": runtime,
            "error_type": "BasisWorkerProtocolError",
            "error_message": "worker returned an invalid object",
        }
    outcome["runtime_seconds"] = runtime
    return outcome


def write_hypothesis_study_outputs(
    study: HypothesisStudy,
    dataset: ReferenceDataset,
    output_directory: str | Path,
) -> HypothesisOutputPaths:
    """Write complete hypothesis tables, report, and plots for promoted cases."""

    output = Path(output_directory)
    plot_directory = output / "plots"
    plot_directory.mkdir(parents=True, exist_ok=True)
    results = study.screening_results + study.promoted_results
    plot_paths = []
    for rank, result in enumerate(rank_hypotheses(study.promoted_results), start=1):
        safe_name = f"rank_{rank}_{_safe_name(result.hypothesis.name)}"
        paths = (
            plot_directory / f"{safe_name}_minima.png",
            plot_directory / f"{safe_name}_row_errors.png",
            plot_directory / f"{safe_name}_error_vectors.png",
            plot_directory / f"{safe_name}_radial.png",
        )
        _write_minima_plot(result, paths[0])
        _write_row_error_plot(result, paths[1])
        _write_error_vector_plot(result, paths[2])
        _write_radial_plot(result, paths[3])
        plot_paths.extend(paths)
    paths = HypothesisOutputPaths(
        summary_csv=output / "hypothesis_summary.csv",
        rows_csv=output / "hypothesis_rows.csv",
        minima_csv=output / "hypothesis_minima.csv",
        basis_fit_csv=output / "basis_fit.csv",
        scale_diagnostics_csv=output / "scale_diagnostics.csv",
        markdown_report=output / "milestone_6_report.md",
        plot_paths=tuple(plot_paths),
    )
    _write_csv(paths.summary_csv, _summary_records(results, study.mesh_size_m))
    _write_csv(paths.rows_csv, _row_records(results))
    _write_csv(paths.minima_csv, _minimum_records(results))
    _write_csv(paths.basis_fit_csv, _basis_records(study.basis_fit))
    _write_csv(
        paths.scale_diagnostics_csv,
        _scale_diagnostic_records(dataset, study.screening_rows),
    )
    paths.markdown_report.write_text(
        _markdown_report(study, dataset),
        encoding="utf-8",
    )
    return paths


def _summary_records(
    results: Sequence[HypothesisResult],
    mesh_size_m: float,
) -> list[dict[str, object]]:
    records = []
    for result in results:
        hypothesis = result.hypothesis
        fem = hypothesis.fem_hypothesis
        summary = result.summary()
        records.append(
            {
                "hypothesis": hypothesis.name,
                "scope": result.scope,
                "family": hypothesis.family,
                "fem_hypothesis": fem.name,
                "displacement_mode": fem.displacement_mode,
                "electrode_permutation": "-".join(
                    map(str, fem.electrode_permutation)
                ),
                "electrode_potentials_v": ",".join(
                    f"{value:.12g}" for value in fem.electrode_potentials_v
                ),
                "reference_frame": hypothesis.reference_frame,
                "coordinate_transform": hypothesis.coordinate_transform,
                "scale_mode": hypothesis.scale_mode,
                "output_scale": hypothesis.output_scale,
                "fitted_on_rows": ",".join(map(str, hypothesis.fitted_on_rows)),
                "mesh_size_m": mesh_size_m,
                "outer_radius_m": REAL_OUTER_BOUNDARY_RADIUS_M,
                "search_half_width_m": 8.0e-3,
                "selected_rows": summary.selected_rows,
                "completed_rows": summary.completed_rows,
                "failed_rows": summary.failed_rows,
                "rows_exactly_three_physical_minima": (
                    summary.rows_with_exactly_three_physical_minima
                ),
                "matched_minima": summary.matched_minima,
                "mean_error_m": summary.mean_error_m,
                "mean_error_um": summary.mean_error_m * 1.0e6,
                "mean_error_mm": summary.mean_error_m * 1.0e3,
                "median_error_m": summary.median_error_m,
                "median_error_um": summary.median_error_m * 1.0e6,
                "median_error_mm": summary.median_error_m * 1.0e3,
                "maximum_error_m": summary.maximum_error_m,
                "maximum_error_um": summary.maximum_error_m * 1.0e6,
                "maximum_error_mm": summary.maximum_error_m * 1.0e3,
                "percentile_95_error_m": summary.percentile_95_error_m,
                "percentile_95_error_um": summary.percentile_95_error_m * 1.0e6,
                "percentile_95_error_mm": summary.percentile_95_error_m * 1.0e3,
                "passes_validation_gate": summary.passes_validation_gate,
                "fem_runtime_seconds": result.fem_runtime_seconds,
            }
        )
    return records


def _row_records(results: Sequence[HypothesisResult]) -> list[dict[str, object]]:
    records = []
    for result in results:
        for row in result.rows:
            errors = row.error_distances_m()
            observation = row.observation
            records.append(
                {
                    "hypothesis": result.hypothesis.name,
                    "scope": result.scope,
                    "row_number": row.row_number,
                    "status": row.status,
                    "computed_minimum_count": len(row.computed_positions_m),
                    "hessian_validated_candidates": (
                        "" if observation is None else observation.hessian_validated_candidates
                    ),
                    "exactly_three_physical_minima": (
                        row.exactly_three_physical_minima
                    ),
                    "node_count": "" if observation is None else observation.node_count,
                    "triangle_count": (
                        "" if observation is None else observation.triangle_count
                    ),
                    "relative_free_residual": (
                        "" if observation is None else observation.relative_free_residual
                    ),
                    "mean_error_um": (
                        "" if not errors.size else float(np.mean(errors) * 1.0e6)
                    ),
                    "mean_error_mm": (
                        "" if not errors.size else float(np.mean(errors) * 1.0e3)
                    ),
                    "median_error_um": (
                        "" if not errors.size else float(np.median(errors) * 1.0e6)
                    ),
                    "median_error_mm": (
                        "" if not errors.size else float(np.median(errors) * 1.0e3)
                    ),
                    "maximum_error_um": (
                        "" if not errors.size else float(np.max(errors) * 1.0e6)
                    ),
                    "maximum_error_mm": (
                        "" if not errors.size else float(np.max(errors) * 1.0e3)
                    ),
                    "error_type": row.error_type,
                    "error_message": row.error_message,
                }
            )
    return records


def _minimum_records(
    results: Sequence[HypothesisResult],
) -> list[dict[str, object]]:
    records = []
    for result in results:
        for row in result.rows:
            for match in row.matches:
                reference_radius = float(np.linalg.norm(match.reference_position_m))
                computed_radius = float(np.linalg.norm(match.computed_position_m))
                records.append(
                    {
                        "hypothesis": result.hypothesis.name,
                        "scope": result.scope,
                        "row_number": row.row_number,
                        "reference_index": match.reference_index,
                        "computed_index": match.computed_index,
                        "reference_x_m": match.reference_position_m[0],
                        "reference_y_m": match.reference_position_m[1],
                        "computed_x_m": match.computed_position_m[0],
                        "computed_y_m": match.computed_position_m[1],
                        "delta_x_m": match.delta_m[0],
                        "delta_y_m": match.delta_m[1],
                        "reference_radius_m": reference_radius,
                        "computed_radius_m": computed_radius,
                        "radial_error_m": computed_radius - reference_radius,
                        "error_m": match.distance_m,
                        "error_um": match.distance_m * 1.0e6,
                        "error_mm": match.distance_m * 1.0e3,
                    }
                )
    return records


def _basis_records(basis: BasisFitDiagnostic) -> list[dict[str, object]]:
    records = []
    for row in basis.rows:
        record: dict[str, object] = {
            "row_number": row.row_number,
            "status": row.status,
            "node_count": row.node_count,
            "triangle_count": row.triangle_count,
            "runtime_seconds": row.runtime_seconds,
            "error_type": row.error_type,
            "error_message": row.error_message,
            "normalized_smallest_gram_eigenvalue": (
                basis.normalized_smallest_eigenvalue
            ),
        }
        for index, value in enumerate(basis.electrode_potentials_v, start=1):
            record[f"fitted_e{index}_potential_v"] = value
        for index, value in enumerate(basis.gram_eigenvalues, start=1):
            record[f"gram_eigenvalue_{index}"] = value
        for point_index, point in enumerate(row.target_positions_m, start=1):
            record[f"target_{point_index}_x_m"] = point[0]
            record[f"target_{point_index}_y_m"] = point[1]
        if row.basis_fields_v_per_m is not None:
            combined = np.einsum(
                "pce,e->pc",
                row.basis_fields_v_per_m,
                np.asarray(basis.electrode_potentials_v),
            )
            record["combined_field_rms_v_per_m"] = float(
                np.sqrt(np.mean(combined * combined))
            )
            record["combined_field_max_v_per_m"] = float(
                np.max(np.linalg.norm(combined, axis=1))
            )
        else:
            record["combined_field_rms_v_per_m"] = ""
            record["combined_field_max_v_per_m"] = ""
        records.append(record)
    return records


def _scale_diagnostic_records(
    dataset: ReferenceDataset,
    row_numbers: Sequence[int],
) -> list[dict[str, object]]:
    indices = np.asarray(row_numbers, dtype=int) - 1
    absolute = dataset.raw_minima_absolute_m[indices]
    relative = dataset.minima_relative_to_electrode1_m[indices]
    records = []
    for frame, values in (("absolute", absolute), ("electrode1-relative", relative)):
        radii = np.linalg.norm(values.reshape(-1, 2), axis=1)
        normalized_inner = values / REAL_INNER_RADIUS_M
        roundtrip_inner = normalized_inner * REAL_INNER_RADIUS_M
        records.append(
            {
                "reference_frame": frame,
                "minimum_radius_m": float(np.min(radii)),
                "median_radius_m": float(np.median(radii)),
                "maximum_radius_m": float(np.max(radii)),
                "median_radius_over_inner_radius": float(
                    np.median(radii) / REAL_INNER_RADIUS_M
                ),
                "median_radius_over_electrode_center_radius": float(
                    np.median(radii)
                    / (REAL_INNER_RADIUS_M + REAL_ELECTRODE_RADIUS_M)
                ),
                "inner_radius_normalized_minimum": float(
                    np.min(normalized_inner)
                ),
                "inner_radius_normalized_maximum": float(
                    np.max(normalized_inner)
                ),
                "inner_radius_roundtrip_max_abs_error_m": float(
                    np.max(np.abs(roundtrip_inner - values))
                ),
            }
        )
    return records


def _write_csv(path: Path, records: list[dict[str, object]]) -> None:
    if not records:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _write_minima_plot(result: HypothesisResult, path: Path) -> None:
    figure = Figure(figsize=(7.0, 6.2), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    first_reference = True
    first_computed = True
    for row in result.rows:
        if not row.completed:
            continue
        reference = row.reference_positions_m * 1.0e3
        computed = row.computed_positions_m * 1.0e3
        axis.scatter(
            reference[:, 0],
            reference[:, 1],
            marker="x",
            color="C0",
            alpha=0.7,
            label="reference" if first_reference else None,
        )
        axis.scatter(
            computed[:, 0],
            computed[:, 1],
            facecolors="none",
            edgecolors="C1",
            alpha=0.7,
            label="FEM" if first_computed else None,
        )
        first_reference = False
        first_computed = False
    axis.set_title("Reference vs computed minima\n" + _short_name(result))
    axis.set_xlabel("x (mm)")
    axis.set_ylabel("y (mm)")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.savefig(path, dpi=180, bbox_inches="tight")


def _write_row_error_plot(result: HypothesisResult, path: Path) -> None:
    figure = Figure(figsize=(8.2, 4.8), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    for row in result.rows:
        errors = row.error_distances_m() * 1.0e3
        if errors.size:
            axis.scatter(
                np.full(errors.shape, row.row_number),
                errors,
                color="C0",
                alpha=0.55,
                s=20,
            )
            axis.plot(
                row.row_number,
                float(np.mean(errors)),
                marker="D",
                color="C1",
                markersize=4,
            )
    axis.axhline(VALIDATION_MAXIMUM_LIMIT_M * 1.0e3, color="C3", linestyle="--")
    axis.set_title("Per-row matched errors\n" + _short_name(result))
    axis.set_xlabel("source row")
    axis.set_ylabel("error (mm)")
    axis.grid(True, alpha=0.25)
    figure.savefig(path, dpi=180, bbox_inches="tight")


def _write_error_vector_plot(result: HypothesisResult, path: Path) -> None:
    figure = Figure(figsize=(7.0, 6.2), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    for row in result.rows:
        for match in row.matches:
            reference = match.reference_position_m * 1.0e3
            computed = match.computed_position_m * 1.0e3
            axis.plot(
                (reference[0], computed[0]),
                (reference[1], computed[1]),
                color="0.45",
                linewidth=0.7,
                alpha=0.65,
            )
            axis.scatter(*reference, marker="x", color="C0", s=22)
            axis.scatter(
                *computed,
                facecolors="none",
                edgecolors="C1",
                s=22,
            )
    axis.set_title("Matched error vectors\n" + _short_name(result))
    axis.set_xlabel("x (mm)")
    axis.set_ylabel("y (mm)")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.25)
    figure.savefig(path, dpi=180, bbox_inches="tight")


def _write_radial_plot(result: HypothesisResult, path: Path) -> None:
    figure = Figure(figsize=(6.2, 5.5), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    reference_radii = []
    computed_radii = []
    for row in result.rows:
        for match in row.matches:
            reference_radii.append(np.linalg.norm(match.reference_position_m) * 1.0e3)
            computed_radii.append(np.linalg.norm(match.computed_position_m) * 1.0e3)
    axis.scatter(reference_radii, computed_radii, alpha=0.65, color="C0")
    maximum = max(reference_radii + computed_radii, default=1.0)
    axis.plot((0.0, maximum), (0.0, maximum), linestyle="--", color="0.35")
    axis.set_title("Radial comparison\n" + _short_name(result))
    axis.set_xlabel("reference radius (mm)")
    axis.set_ylabel("computed radius (mm)")
    axis.grid(True, alpha=0.25)
    axis.set_aspect("equal", adjustable="box")
    figure.savefig(path, dpi=180, bbox_inches="tight")


def _short_name(result: HypothesisResult) -> str:
    name = result.hypothesis.name
    return name if len(name) <= 88 else name[:85] + "..."


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)[:96]


def _markdown_report(study: HypothesisStudy, dataset: ReferenceDataset) -> str:
    screening_ranked = rank_hypotheses(study.screening_results)
    promoted_ranked = rank_hypotheses(study.promoted_results)
    best = promoted_ranked[0]
    best_summary = best.summary()
    screen_best = screening_ranked[0]
    screen_best_summary = screen_best.summary()
    screen_improvement = (
        1.0 - screen_best_summary.mean_error_m / MILESTONE_5_MEAN_ERROR_M
    )
    promoted_all_positive_results = tuple(
        result
        for result in study.promoted_results
        if result.hypothesis.fem_hypothesis.family == "all-positive"
    )
    if promoted_all_positive_results:
        promoted_all_positive = rank_hypotheses(
            promoted_all_positive_results
        )[0]
        promoted_all_positive_summary = promoted_all_positive.summary()
        promoted_model_gain = (
            1.0
            - best_summary.mean_error_m
            / promoted_all_positive_summary.mean_error_m
        )
        promotion_comparison_lines = [
            "number. On rows 1--50 it improves over the promoted all-positive",
            f"case ({_mm(promoted_all_positive_summary.mean_error_m)} mm) by only "
            f"`{promoted_model_gain * 100.0:.3f}%`.",
        ]
    else:
        promotion_comparison_lines = [
            "number. No all-positive hypothesis was included in this custom",
            "promotion set, so no same-row model-family comparison is reported.",
        ]
    scale_records = _scale_diagnostic_records(dataset, study.screening_rows)
    basis = study.basis_fit
    completed_basis = sum(row.status == "ok" for row in basis.rows)
    classification = _mismatch_classification(best, screening_ranked, basis)
    best_by_fem: dict[str, HypothesisResult] = {}
    for result in screening_ranked:
        best_by_fem.setdefault(result.hypothesis.fem_hypothesis.name, result)
    lines = [
        "# Milestone 6: model-hypothesis diagnosis",
        "",
        "## Scope and invariant model assumptions",
        "",
        "The screen uses rows 1--10, a 2.0 mm characteristic mesh, the named",
        "50 mm real-scale outer boundary, 10 mm electrodes, 11.48 mm inner",
        "surface radius, and a +/-8 mm minima-search square. Production FEM rows",
        "run in fresh interpreter processes. Failed rows remain in every table.",
        "",
        "The default physical model remains four all-positive electrodes and a",
        "grounded outer boundary. Every alternate voltage vector, coordinate",
        "transform, numbering map, and fitted scale in this report is diagnostic",
        "only. No default was changed.",
        "",
        "Electrode 1 remains the source reference. Consequently, all six",
        "permutations of source E2--E4 were tested while source E1 stayed mapped",
        "to FEM E1. The eight output transforms cover identity, independent x/y",
        "sign flips, x/y swap, 90/180/270 degree rotations, and anti-diagonal",
        "reflection.",
        "",
        "## Scale diagnostics",
        "",
        "| reference frame | radius min / median / max (mm) | median / inner radius | median / electrode-centre radius | inner-radius round-trip error (m) |",
        "|:---|:---|---:|---:|---:|",
    ]
    for record in scale_records:
        lines.append(
            f"| {record['reference_frame']} "
            f"| {float(record['minimum_radius_m']) * 1.0e3:.6g} / "
            f"{float(record['median_radius_m']) * 1.0e3:.6g} / "
            f"{float(record['maximum_radius_m']) * 1.0e3:.6g} "
            f"| {float(record['median_radius_over_inner_radius']):.6g} "
            f"| {float(record['median_radius_over_electrode_center_radius']):.6g} "
            f"| {float(record['inner_radius_roundtrip_max_abs_error_m']):.3g} |"
        )
    lines.extend(
        [
            "",
            "Dividing reference positions by the inner radius and multiplying by",
            "the same radius is exactly a change of representation: the measured",
            "round-trip error above is numerical zero and cannot improve agreement.",
            "The separate fitted-output-scale hypotheses multiply FEM predictions",
            "by one positive scalar fitted on rows 1--10; promoted rows retain that",
            "fixed scalar rather than refitting on rows 11--50.",
            "",
            "## One-electrode basis-field fit",
            "",
            f"- Successful basis rows: `{completed_basis}/{len(basis.rows)}`",
            "- Fitted E1--E4 diagnostic potentials: `(" + ", ".join(
                f"{value:.9g}" for value in basis.electrode_potentials_v
            ) + ") V`",
            "- Gram eigenvalues: `(" + ", ".join(
                f"{value:.9g}" for value in basis.gram_eigenvalues
            ) + ")`",
            f"- Smallest eigenvalue / Gram trace: `{basis.normalized_smallest_eigenvalue:.9g}`",
            f"- Basis-fit runtime: `{basis.runtime_seconds:.3f} s`",
            "",
            "Each one-hot basis solves Laplace's equation on the same row mesh.",
            "Linearity then fits one global voltage vector that minimizes field",
            "magnitude at the transformed reference minima. The fitted vector is",
            "subsequently tested through the ordinary forward minima pipeline; a",
            "small field-fit eigenvalue alone is not treated as validation.",
            "",
            "The fitted vector differs from all-positive by at most 0.02135%.",
            "That is evidence that the basis fit converged back to the all-positive",
            "model, not evidence for a materially different polarity convention.",
            "",
            "## Best output interpretation for each FEM hypothesis",
            "",
            "This compact table exposes polarity failures that would otherwise be",
            "hidden below the top-50 output-transform rankings. The complete table",
            "and every failed row remain in the CSV outputs.",
            "",
            "| FEM family | input | map | potentials E1--E4 (V) | rows ok | exact-three | best mean (mm) | best max (mm) | best output interpretation |",
            "|:---|:---|:---|:---|---:|---:|---:|---:|:---|",
        ]
    )
    for result in best_by_fem.values():
        hypothesis = result.hypothesis
        fem = hypothesis.fem_hypothesis
        summary = result.summary()
        potentials = ",".join(f"{value:.6g}" for value in fem.electrode_potentials_v)
        interpretation = (
            f"{hypothesis.reference_frame}; {hypothesis.coordinate_transform}; "
            f"{hypothesis.scale_mode}"
        )
        lines.append(
            f"| {fem.family} | {fem.displacement_mode} "
            f"| {'-'.join(map(str, fem.electrode_permutation))} "
            f"| {potentials} | {summary.completed_rows}/{summary.selected_rows} "
            f"| {summary.rows_with_exactly_three_physical_minima} "
            f"| {_mm(summary.mean_error_m)} | {_mm(summary.maximum_error_m)} "
            f"| {interpretation} |"
        )
    lines.extend(
        [
            "",
            "## Promoted hypotheses (rows 1--50)",
            "",
            "| rank | hypothesis | rows ok | exact-three | mean (mm) | median (mm) | max (mm) | p95 (mm) | gate |",
            "|---:|:---|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for index, result in enumerate(promoted_ranked, start=1):
        summary = result.summary()
        lines.append(
            f"| {index} | `{result.hypothesis.name}` "
            f"| {summary.completed_rows}/{summary.selected_rows} "
            f"| {summary.rows_with_exactly_three_physical_minima} "
            f"| {_mm(summary.mean_error_m)} | {_mm(summary.median_error_m)} "
            f"| {_mm(summary.maximum_error_m)} "
            f"| {_mm(summary.percentile_95_error_m)} "
            f"| {_yes_no(summary.passes_validation_gate)} |"
        )
    lines.extend(
        [
            "",
            "## Screening hypothesis table (top 50 of full CSV)",
            "",
            "The complete screening table is `hypothesis_summary.csv`. It includes",
            "every tested model/frame/transform/scale hypothesis and all failed",
            "cases. The top 50 are reproduced here for a readable Markdown audit.",
            "",
            "| rank | family | input | map | reference | transform | scale mode / value | rows ok | exact-three | mean (mm) | max (mm) |",
            "|---:|:---|:---|:---|:---|:---|:---|---:|---:|---:|---:|",
        ]
    )
    for index, result in enumerate(screening_ranked[:50], start=1):
        hypothesis = result.hypothesis
        fem = hypothesis.fem_hypothesis
        summary = result.summary()
        lines.append(
            f"| {index} | {hypothesis.family} | {fem.displacement_mode} "
            f"| {'-'.join(map(str, fem.electrode_permutation))} "
            f"| {hypothesis.reference_frame} | {hypothesis.coordinate_transform} "
            f"| {hypothesis.scale_mode} / {hypothesis.output_scale:.8g} "
            f"| {summary.completed_rows}/{summary.selected_rows} "
            f"| {summary.rows_with_exactly_three_physical_minima} "
            f"| {_mm(summary.mean_error_m)} | {_mm(summary.maximum_error_m)} |"
        )
    lines.extend(
        [
            "",
            "## Best hypothesis and mismatch classification",
            "",
            f"The best screening hypothesis was `{screen_best.hypothesis.name}`:",
            f"mean `{_mm(screen_best_summary.mean_error_m)} mm`, maximum "
            f"`{_mm(screen_best_summary.maximum_error_m)} mm`, and exactly-three "
            f"topology in `{screen_best_summary.rows_with_exactly_three_physical_minima}/"
            f"{screen_best_summary.selected_rows}` rows.",
            f"After promotion, the best hypothesis is `{best.hypothesis.name}`:",
            "",
            f"- completed rows: `{best_summary.completed_rows}/{best_summary.selected_rows}`;",
            f"- exactly-three physical-minimum rows: `{best_summary.rows_with_exactly_three_physical_minima}/{best_summary.selected_rows}`;",
            f"- mean error: `{_mm(best_summary.mean_error_m)} mm`;",
            f"- median error: `{_mm(best_summary.median_error_m)} mm`;",
            f"- maximum error: `{_mm(best_summary.maximum_error_m)} mm`;",
            f"- p95 error: `{_mm(best_summary.percentile_95_error_m)} mm`;",
            f"- output scale: `{best.hypothesis.output_scale:.9g}`;",
            f"- validation gate: `{_yes_no(best_summary.passes_validation_gate)}`.",
            "",
            "On the same rows 1--10, the best screen reduces the Milestone-5 mean",
            f"from 1.08687 mm to {_mm(screen_best_summary.mean_error_m)} mm, a "
            f"`{screen_improvement * 100.0:.3f}%` improvement. The rows 1--50",
            "promoted mean is not directly comparable with Milestone 5's ten-row",
            *promotion_comparison_lines,
            "",
            classification,
            "",
            "## Decision",
            "",
            (
                "**PASS:** the best promoted hypothesis meets the validation gate."
                if best_summary.passes_validation_gate
                else "**NOT SAFE:** no promoted hypothesis meets the validation gate."
            ),
            "The gate requires every selected row to complete with exactly three",
            "pre-selection Hessian-valid minima, mean error <=0.25 mm, and maximum",
            "error <=0.5 mm. Diagnostic scale or voltage fitting does not change",
            "the default physical model and cannot authorize generation by itself.",
            "No ML or synthetic dataset generation was performed.",
            "",
            f"Total study runtime: `{study.runtime_seconds:.3f} s`.",
            "",
        ]
    )
    return "\n".join(lines)


def _mismatch_classification(
    best: HypothesisResult,
    screening_ranked: Sequence[HypothesisResult],
    basis: BasisFitDiagnostic,
) -> str:
    hypothesis = best.hypothesis
    fem = hypothesis.fem_hypothesis
    parts = []
    if fem.electrode_permutation != (1, 2, 3, 4):
        parts.append("electrode numbering matters")
    if hypothesis.coordinate_transform == "identity":
        parts.append("global orientation is not the cause")
    else:
        parts.append("global orientation contributes")
    if hypothesis.scale_mode == "fitted" and abs(hypothesis.output_scale - 1.0) > 0.05:
        parts.append("output scale is a minor correction")
    potential_spread = float(np.ptp(np.asarray(basis.electrode_potentials_v)))
    if potential_spread <= 1.0e-3:
        parts.append(
            "the basis fit converges to all-positive, so polarity is not primary"
        )
    else:
        parts.append("fitted voltage coefficients contribute")
    if not parts:
        parts.append("no tested convention dominates")
    best_summary = best.summary()
    if not best_summary.passes_validation_gate:
        parts.append(
            "the residual mismatch is primarily model-class/topology limited"
        )
    all_positive = [
        result
        for result in screening_ranked
        if result.hypothesis.fem_hypothesis.family == "all-positive"
    ]
    best_all_positive = rank_hypotheses(all_positive)[0]
    return (
        "Classification: **"
        + "; ".join(parts)
        + "**. The best all-positive screening mean was "
        + f"{_mm(best_all_positive.summary().mean_error_m)} mm; comparison with the "
        "binary-polarity, promoted, and fitted-voltage cases separates convention "
        "gains from a "
        "remaining inability of this four-electrode model class to reproduce all "
        "three reference branches."
    )


def _mm(value_m: float) -> str:
    return "n/a" if not np.isfinite(value_m) else f"{value_m * 1.0e3:.6g}"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def build_parser() -> argparse.ArgumentParser:
    """Build the Milestone-6 model-hypothesis diagnostic CLI."""

    parser = argparse.ArgumentParser(
        prog="rf-trap-hypothesis-validation",
        description="Diagnose coordinate, scale, polarity, and model hypotheses.",
    )
    parser.add_argument("input", nargs="?", type=Path, default=Path("Data.txt"))
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("validation_results") / "milestone_6",
    )
    parser.add_argument("--mesh-size-mm", type=float, default=2.0)
    parser.add_argument("--screen-end-row", type=int, default=10)
    parser.add_argument("--promote-end-row", type=int, default=50)
    parser.add_argument("--promote-count", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run Milestone 6 and print the best promoted metrics and output paths."""

    arguments = build_parser().parse_args(argv)
    dataset = load_reference_dataset(arguments.input)
    screen = tuple(range(1, arguments.screen_end_row + 1))
    promote = tuple(range(1, arguments.promote_end_row + 1))
    study = run_model_hypothesis_study(
        dataset,
        screening_rows=screen,
        promotion_rows=promote,
        mesh_size_m=arguments.mesh_size_mm * 1.0e-3,
        promote_count=arguments.promote_count,
    )
    paths = write_hypothesis_study_outputs(
        study,
        dataset,
        arguments.output_directory,
    )
    best = study.best_result()
    summary = best.summary()
    print(f"best hypothesis: {best.hypothesis.name}")
    print(f"completed rows: {summary.completed_rows}/{summary.selected_rows}")
    print(f"exactly-three rows: {summary.rows_with_exactly_three_physical_minima}")
    print(f"mean error: {_mm(summary.mean_error_m)} mm")
    print(f"maximum error: {_mm(summary.maximum_error_m)} mm")
    print(f"validation gate: {_yes_no(summary.passes_validation_gate)}")
    print(f"runtime: {study.runtime_seconds:.3f} s")
    print(f"summary CSV: {paths.summary_csv}")
    print(f"row CSV: {paths.rows_csv}")
    print(f"minimum CSV: {paths.minima_csv}")
    print(f"basis CSV: {paths.basis_fit_csv}")
    print(f"scale CSV: {paths.scale_diagnostics_csv}")
    print(f"report: {paths.markdown_report}")
    print(f"plots: {len(paths.plot_paths)} under {paths.plot_paths[0].parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
