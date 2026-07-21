"""Milestone 9 targeted refinement and calibrated forward-model diagnostics.

Every model change in this module is an explicitly named diagnostic. The
legacy all-positive, real-scale forward configuration remains unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import pickle
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from math import cos, radians, sin, sqrt
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from .config import ForwardModelConfig, GeometryConfig
from .dataset import ReferenceDataset, load_reference_dataset
from .hypothesis_validation import fit_basis_potentials
from .mesh import estimate_central_triangle_count
from .minima import LocalMinimum
from .minima_modes import MinimaModeResult, RobustMinimaConfig
from .real_scale import (
    REAL_ELECTRODE_RADIUS_M,
    REAL_INNER_RADIUS_M,
    REAL_OUTER_BOUNDARY_RADIUS_M,
    locally_refined_real_scale_forward_config,
)
from .reference_validation import (
    MinimumMatch,
    ReferenceValidationVariant,
    match_minima_by_distance,
    prepare_reference_row_inputs,
)

MILESTONE_8_MEAN_ERROR_M = 1.08754e-3
VALIDATION_MEAN_LIMIT_M = 0.25e-3
VALIDATION_MAXIMUM_LIMIT_M = 0.50e-3
BEST_KNOWN_MAPPING = (1, 3, 2, 4)
DEFAULT_REFINEMENT_SIZES_M = (
    500.0e-6,
    200.0e-6,
    100.0e-6,
    50.0e-6,
    20.0e-6,
    10.0e-6,
    5.0e-6,
    1.0e-6,
    10.0 ** -6.5,
)
MILESTONE_6_FITTED_VOLTAGES = (
    0.999926798,
    0.999809418,
    0.999786521,
    1.0,
)


@dataclass(frozen=True)
class GeometryVariant:
    """One explicit diagonal four-electrode geometry hypothesis."""

    name: str
    electrode_center_radius_m: float
    electrode_radius_m: float
    outer_boundary_radius_m: float
    interpretation: str = "center-radius"

    def __post_init__(self) -> None:
        """Validate non-overlap and outer-domain containment analytically."""

        values = np.asarray(
            (
                self.electrode_center_radius_m,
                self.electrode_radius_m,
                self.outer_boundary_radius_m,
            ),
            dtype=float,
        )
        if not self.name or not self.interpretation:
            raise ValueError("geometry variant names must not be empty")
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("geometry radii must be finite and positive")
        nearest_center_separation = sqrt(2.0) * self.electrode_center_radius_m
        if nearest_center_separation <= 2.0 * self.electrode_radius_m:
            raise ValueError("geometry variant has touching or overlapping electrodes")
        if (
            self.electrode_center_radius_m + self.electrode_radius_m
            >= self.outer_boundary_radius_m
        ):
            raise ValueError("geometry variant leaves the outer domain")

    @property
    def surface_clearance_m(self) -> float:
        """Return center-to-nearest-electrode-surface clearance."""

        return self.electrode_center_radius_m - self.electrode_radius_m

    def geometry_config(
        self,
        voltage_model: VoltageModel | None = None,
    ) -> GeometryConfig:
        """Build the immutable diagonal geometry for one voltage hypothesis."""

        voltage = voltage_model or VoltageModel("all-positive", (1.0,) * 4)
        coordinate = self.electrode_center_radius_m / sqrt(2.0)
        return GeometryConfig(
            electrode_radius_m=self.electrode_radius_m,
            nominal_centers_m=(
                (-coordinate, +coordinate),
                (+coordinate, +coordinate),
                (-coordinate, -coordinate),
                (+coordinate, -coordinate),
            ),
            outer_radius_m=self.outer_boundary_radius_m,
            electrode_potentials_v=voltage.electrode_potentials_v,
            outer_potential_v=voltage.outer_potential_v,
        )


@dataclass(frozen=True)
class VoltageModel:
    """One named four-electrode plus outer-boundary voltage hypothesis."""

    name: str
    electrode_potentials_v: tuple[float, float, float, float]
    outer_potential_v: float = 0.0
    origin: str = "specified"

    def __post_init__(self) -> None:
        """Validate a finite, nonconstant boundary-voltage model."""

        values = np.asarray((*self.electrode_potentials_v, self.outer_potential_v))
        if not self.name or not self.origin:
            raise ValueError("voltage model names must not be empty")
        if values.shape != (5,) or not np.all(np.isfinite(values)):
            raise ValueError("voltage model must contain five finite values")
        if np.ptp(values) <= np.finfo(float).eps:
            raise ValueError("boundary potentials must produce a nonzero field")


@dataclass(frozen=True)
class OutputTransform:
    """Diagnostic global scale, rotation, and area-preserving anisotropy."""

    global_scale: float = 1.0
    rotation_deg: float = 0.0
    anisotropy_ratio: float = 1.0

    def __post_init__(self) -> None:
        """Enforce the documented calibration bounds."""

        values = np.asarray(
            (self.global_scale, self.rotation_deg, self.anisotropy_ratio),
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("output-transform parameters must be finite")
        if not 0.7 <= self.global_scale <= 1.3:
            raise ValueError("global output scale must lie in [0.7, 1.3]")
        if not -15.0 <= self.rotation_deg <= 15.0:
            raise ValueError("rotation must lie in [-15, 15] degrees")
        if not 0.85 <= self.anisotropy_ratio <= 1.15:
            raise ValueError("anisotropy ratio must lie in [0.85, 1.15]")

    @property
    def x_scale(self) -> float:
        """Return the total x-axis output scale before rotation."""

        return self.global_scale * sqrt(self.anisotropy_ratio)

    @property
    def y_scale(self) -> float:
        """Return the total y-axis output scale before rotation."""

        return self.global_scale / sqrt(self.anisotropy_ratio)


@dataclass(frozen=True)
class CalibrationCase:
    """One fully named FEM and output-transform hypothesis."""

    name: str
    family: str
    geometry: GeometryVariant
    voltage: VoltageModel
    electrode_mapping: tuple[int, int, int, int] = BEST_KNOWN_MAPPING
    output_transform: OutputTransform = OutputTransform()

    def __post_init__(self) -> None:
        """Validate the fixed-reference-electrode permutation."""

        if not self.name or not self.family:
            raise ValueError("calibration case names must not be empty")
        ReferenceValidationVariant(
            name=self.name,
            electrode_permutation=self.electrode_mapping,
        )


@dataclass(frozen=True)
class CalibrationRow:
    """One row-level robust FEM result and reference assignment."""

    hypothesis_name: str
    family: str
    scope: str
    row_number: int
    status: str
    reference_positions_m: NDArray[np.float64]
    raw_computed_positions_m: NDArray[np.float64]
    computed_positions_m: NDArray[np.float64]
    matches: tuple[MinimumMatch, ...]
    topology_candidate_count: int
    selected_interpolation_sensitive: int
    rejected_candidates: int
    total_candidates: int
    node_count: int
    triangle_count: int
    relative_free_residual: float
    runtime_seconds: float
    error_type: str = ""
    error_message: str = ""

    @property
    def completed(self) -> bool:
        """Return whether exactly three transformed outputs were matched."""

        return self.status == "ok" and len(self.matches) == 3

    @property
    def exactly_three_topology(self) -> bool:
        """Return whether robust mode accepted exactly three candidates."""

        return self.completed and self.topology_candidate_count == 3

    def errors_m(self) -> NDArray[np.float64]:
        """Return the three assignment errors in metres."""

        return np.asarray([item.distance_m for item in self.matches], dtype=float)


@dataclass(frozen=True)
class CalibrationSummary:
    """Aggregate ranking metrics and complete model parameters."""

    hypothesis_name: str
    family: str
    scope: str
    geometry_name: str
    voltage_name: str
    electrode_mapping: tuple[int, int, int, int]
    electrode_center_radius_m: float
    electrode_radius_m: float
    outer_boundary_radius_m: float
    electrode_potentials_v: tuple[float, float, float, float]
    outer_potential_v: float
    output_transform: OutputTransform
    central_mesh_size_m: float
    selected_rows: int
    completed_rows: int
    exactly_three_rows: int
    matched_minima: int
    mean_error_m: float
    median_error_m: float
    maximum_error_m: float
    percentile_95_error_m: float
    selected_interpolation_sensitive: int
    rejected_candidates: int
    runtime_seconds: float
    validation_gate_passed: bool


@dataclass(frozen=True)
class CalibrationEvaluation:
    """One summary plus the underlying row-level evidence."""

    case: CalibrationCase
    scope: str
    central_mesh_size_m: float
    rows: tuple[CalibrationRow, ...]

    def summary(self) -> CalibrationSummary:
        """Aggregate this evaluation without omitting failed rows."""

        return summarize_calibration_rows(
            self.rows,
            self.case,
            central_mesh_size_m=self.central_mesh_size_m,
        )


@dataclass(frozen=True)
class LocalRefinementRecord:
    """One row at one requested central mesh size, including skipped cases."""

    central_mesh_size_m: float
    estimated_central_triangles: int
    row_number: int
    status: str
    node_count: int = 0
    triangle_count: int = 0
    runtime_seconds: float = 0.0
    exactly_three_robust_minima: bool = False
    topology_candidate_count: int = 0
    selected_positions_m: NDArray[np.float64] | None = None
    branch_shift_m: float = float("nan")
    mean_reference_error_m: float = float("nan")
    maximum_reference_error_m: float = float("nan")
    rejected_candidates: int = 0
    flagged_candidates: int = 0
    skip_or_failure_reason: str = ""


@dataclass(frozen=True)
class Milestone9Study:
    """All refinement, calibration, promotion, and runtime evidence."""

    local_refinement: tuple[LocalRefinementRecord, ...]
    geometry_evaluations: tuple[CalibrationEvaluation, ...]
    voltage_evaluations: tuple[CalibrationEvaluation, ...]
    combined_evaluations: tuple[CalibrationEvaluation, ...]
    promoted_evaluations: tuple[CalibrationEvaluation, ...]
    chosen_central_mesh_size_m: float
    local_refinement_improvement_fraction: float
    fitted_voltage: VoltageModel
    runtime_seconds: float
    resumed_from_interrupted_local_run: bool = False


@dataclass(frozen=True)
class Milestone9OutputPaths:
    """Required tables, report, and plot paths written for Milestone 9."""

    local_csv: Path
    local_report_markdown: Path
    geometry_csv: Path
    voltage_csv: Path
    combined_csv: Path
    per_row_csv: Path
    per_minimum_csv: Path
    report_markdown: Path
    plot_paths: tuple[Path, ...]


def default_geometry_variant() -> GeometryVariant:
    """Return the supplied real-scale geometry as an explicit variant."""

    return GeometryVariant(
        name="real-scale-default",
        electrode_center_radius_m=REAL_INNER_RADIUS_M + REAL_ELECTRODE_RADIUS_M,
        electrode_radius_m=REAL_ELECTRODE_RADIUS_M,
        outer_boundary_radius_m=REAL_OUTER_BOUNDARY_RADIUS_M,
        interpretation="inner-radius-is-surface-clearance",
    )


def generate_geometry_variants() -> tuple[GeometryVariant, ...]:
    """Return a curated one-factor scan spanning every requested coarse range."""

    base = default_geometry_variant()
    variants = [base]
    for center_mm in (15.0, 18.0, 20.0, 23.0, 25.0):
        try:
            variants.append(
                GeometryVariant(
                    name=f"center-{center_mm:g}mm",
                    electrode_center_radius_m=center_mm * 1.0e-3,
                    electrode_radius_m=base.electrode_radius_m,
                    outer_boundary_radius_m=base.outer_boundary_radius_m,
                )
            )
        except ValueError:
            # The 15 mm, 10 mm-radius hypothesis geometrically overlaps and is
            # scientifically excluded rather than passed to Gmsh.
            continue
    for electrode_mm in (8.0, 9.0, 11.0, 12.0):
        variants.append(
            GeometryVariant(
                name=f"electrode-radius-{electrode_mm:g}mm",
                electrode_center_radius_m=base.electrode_center_radius_m,
                electrode_radius_m=electrode_mm * 1.0e-3,
                outer_boundary_radius_m=base.outer_boundary_radius_m,
            )
        )
    for outer_mm in (30.0, 40.0, 65.0, 80.0):
        try:
            variants.append(
                GeometryVariant(
                    name=f"outer-{outer_mm:g}mm",
                    electrode_center_radius_m=base.electrode_center_radius_m,
                    electrode_radius_m=base.electrode_radius_m,
                    outer_boundary_radius_m=outer_mm * 1.0e-3,
                )
            )
        except ValueError:
            continue
    variants.append(
        GeometryVariant(
            name="outer-30mm-scaled-safe",
            electrode_center_radius_m=18.0e-3,
            electrode_radius_m=8.0e-3,
            outer_boundary_radius_m=30.0e-3,
            interpretation="outer-30mm-requires-nonoverlapping-inner-scale",
        )
    )
    variants.append(
        GeometryVariant(
            name="inner-plus-radius-used-as-diagonal-coordinate",
            electrode_center_radius_m=sqrt(2.0)
            * (REAL_INNER_RADIUS_M + REAL_ELECTRODE_RADIUS_M),
            electrode_radius_m=REAL_ELECTRODE_RADIUS_M,
            outer_boundary_radius_m=REAL_OUTER_BOUNDARY_RADIUS_M,
            interpretation="21.48mm-treated-as-diagonal-coordinate",
        )
    )
    return tuple(variants)


def normalize_voltage_vector(
    electrode_potentials_v: ArrayLike,
    *,
    outer_potential_v: float = 0.0,
    remove_global_offset: bool = False,
) -> tuple[tuple[float, float, float, float], float]:
    """Normalize boundary differences to unit maximum absolute magnitude.

    When ``remove_global_offset`` is true, the electrode mean is subtracted
    from *both* electrode and outer-boundary values. This is a gauge shift and
    therefore preserves the electric field. Subtracting only the electrode
    mean while holding the outer boundary at zero would be a different model.
    """

    electrodes = np.asarray(electrode_potentials_v, dtype=float)
    if electrodes.shape != (4,) or not np.all(np.isfinite(electrodes)):
        raise ValueError("electrode_potentials_v must contain four finite values")
    if not np.isfinite(outer_potential_v):
        raise ValueError("outer_potential_v must be finite")
    outer = float(outer_potential_v)
    if remove_global_offset:
        offset = float(np.mean(electrodes))
        electrodes = electrodes - offset
        outer -= offset
    differences = np.concatenate((electrodes - outer, (0.0,)))
    amplitude = float(np.max(np.abs(differences)))
    if amplitude <= np.finfo(float).eps:
        raise ValueError("voltage vector is constant relative to the outer boundary")
    electrodes /= amplitude
    outer /= amplitude
    return tuple(float(item) for item in electrodes), float(outer)


def voltage_model_variants(
    fitted_voltage: Sequence[float] = MILESTONE_6_FITTED_VOLTAGES,
) -> tuple[VoltageModel, ...]:
    """Return default, checkerboard, fitted, and gauge-equivalent variants."""

    fitted, fitted_outer = normalize_voltage_vector(fitted_voltage)
    shifted, shifted_outer = normalize_voltage_vector(
        fitted_voltage,
        remove_global_offset=True,
    )
    return (
        VoltageModel("all-positive", (1.0, 1.0, 1.0, 1.0), origin="default"),
        VoltageModel(
            "checkerboard",
            (1.0, -1.0, -1.0, 1.0),
            origin="named-diagnostic",
        ),
        VoltageModel(
            "basis-fit-milestone-6",
            fitted,
            fitted_outer,
            origin="one-electrode-basis-fit-rows-1-10-h2mm",
        ),
        VoltageModel(
            "basis-fit-milestone-6-gauge-shifted",
            shifted,
            shifted_outer,
            origin="gauge-equivalent-offset-removal",
        ),
    )


def apply_output_transform(
    positions_m: ArrayLike,
    transform: OutputTransform,
) -> NDArray[np.float64]:
    """Apply anisotropic scaling followed by a rigid rotation."""

    positions = np.asarray(positions_m, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2 or not np.all(np.isfinite(positions)):
        raise ValueError("positions_m must have shape (n, 2) and be finite")
    scaled = positions * np.asarray((transform.x_scale, transform.y_scale))
    angle = radians(transform.rotation_deg)
    rotation = np.asarray(((cos(angle), -sin(angle)), (sin(angle), cos(angle))))
    return scaled @ rotation.T


def fit_output_transform(
    computed_sets_m: Sequence[ArrayLike],
    reference_sets_m: Sequence[ArrayLike],
) -> OutputTransform:
    """Fit bounded scale, rotation, and anisotropy using assignment error."""

    if not computed_sets_m or len(computed_sets_m) != len(reference_sets_m):
        raise ValueError("computed and reference set sequences must be equal and nonempty")
    computed = [np.asarray(item, dtype=float) for item in computed_sets_m]
    reference = [np.asarray(item, dtype=float) for item in reference_sets_m]
    if any(a.shape != (3, 2) or b.shape != (3, 2) for a, b in zip(computed, reference, strict=True)):
        raise ValueError("every calibration point set must have shape (3, 2)")

    def objective(parameters: NDArray[np.float64]) -> float:
        transform = OutputTransform(
            global_scale=float(parameters[0]),
            rotation_deg=float(parameters[1]),
            anisotropy_ratio=float(parameters[2]),
        )
        squared = []
        for predicted, target in zip(computed, reference, strict=True):
            transformed = apply_output_transform(predicted, transform)
            matches = match_minima_by_distance(target, transformed)
            squared.extend(item.distance_m**2 for item in matches)
        # Work in squared millimetres so generic optimizer tolerances do not
        # mistake a 0.1 mm residual for a numerically negligible objective.
        return 1.0e6 * float(np.mean(squared))

    result = minimize(
        objective,
        x0=np.asarray((1.0, 0.0, 1.0)),
        method="Powell",
        bounds=((0.7, 1.3), (-15.0, 15.0), (0.85, 1.15)),
        options={"maxiter": 1200, "xtol": 1.0e-11, "ftol": 1.0e-13},
    )
    return OutputTransform(
        global_scale=float(result.x[0]),
        rotation_deg=float(result.x[1]),
        anisotropy_ratio=float(result.x[2]),
    )


def passes_validation_gate(
    *,
    selected_rows: int,
    completed_rows: int,
    exactly_three_rows: int,
    mean_error_m: float,
    maximum_error_m: float,
) -> bool:
    """Apply the stated error and all-rows topology gate."""

    return bool(
        selected_rows > 0
        and completed_rows == selected_rows
        and exactly_three_rows == selected_rows
        and np.isfinite(mean_error_m)
        and np.isfinite(maximum_error_m)
        and mean_error_m <= VALIDATION_MEAN_LIMIT_M
        and maximum_error_m <= VALIDATION_MAXIMUM_LIMIT_M
    )


def summarize_calibration_rows(
    rows: Sequence[CalibrationRow],
    case: CalibrationCase,
    *,
    central_mesh_size_m: float,
) -> CalibrationSummary:
    """Summarize completed matches while preserving failures in gate counts."""

    if not rows:
        raise ValueError("rows must not be empty")
    errors = np.asarray(
        [match.distance_m for row in rows for match in row.matches],
        dtype=float,
    )
    if errors.size:
        mean = float(np.mean(errors))
        median = float(np.median(errors))
        maximum = float(np.max(errors))
        percentile = float(np.percentile(errors, 95.0))
    else:
        mean = median = maximum = percentile = float("nan")
    completed = sum(row.completed for row in rows)
    exact = sum(row.exactly_three_topology for row in rows)
    geometry = case.geometry
    voltage = case.voltage
    return CalibrationSummary(
        hypothesis_name=case.name,
        family=case.family,
        scope=rows[0].scope,
        geometry_name=geometry.name,
        voltage_name=voltage.name,
        electrode_mapping=case.electrode_mapping,
        electrode_center_radius_m=geometry.electrode_center_radius_m,
        electrode_radius_m=geometry.electrode_radius_m,
        outer_boundary_radius_m=geometry.outer_boundary_radius_m,
        electrode_potentials_v=voltage.electrode_potentials_v,
        outer_potential_v=voltage.outer_potential_v,
        output_transform=case.output_transform,
        central_mesh_size_m=central_mesh_size_m,
        selected_rows=len(rows),
        completed_rows=completed,
        exactly_three_rows=exact,
        matched_minima=int(errors.size),
        mean_error_m=mean,
        median_error_m=median,
        maximum_error_m=maximum,
        percentile_95_error_m=percentile,
        selected_interpolation_sensitive=sum(
            row.selected_interpolation_sensitive for row in rows
        ),
        rejected_candidates=sum(row.rejected_candidates for row in rows),
        runtime_seconds=sum(row.runtime_seconds for row in rows),
        validation_gate_passed=passes_validation_gate(
            selected_rows=len(rows),
            completed_rows=completed,
            exactly_three_rows=exact,
            mean_error_m=mean,
            maximum_error_m=maximum,
        ),
    )


def rank_calibration_evaluations(
    evaluations: Sequence[CalibrationEvaluation],
) -> tuple[CalibrationEvaluation, ...]:
    """Rank complete cases by Data.txt error, using topology as a tie-breaker.

    The validation gate still requires exactly-three topology in every row.
    Ranking does not hide a spatially better complete hypothesis merely because
    it has an extra accepted candidate before robust lowest-three selection.
    """

    def key(item: CalibrationEvaluation) -> tuple[float, ...]:
        summary = item.summary()
        finite_mean = summary.mean_error_m if np.isfinite(summary.mean_error_m) else np.inf
        finite_max = summary.maximum_error_m if np.isfinite(summary.maximum_error_m) else np.inf
        finite_p95 = (
            summary.percentile_95_error_m
            if np.isfinite(summary.percentile_95_error_m)
            else np.inf
        )
        return (
            -float(summary.completed_rows),
            finite_mean,
            finite_max,
            finite_p95,
            -float(summary.exactly_three_rows),
            summary.hypothesis_name,
        )

    return tuple(sorted(evaluations, key=key))


def evaluate_calibration_case(
    dataset: ReferenceDataset,
    case: CalibrationCase,
    row_numbers: Sequence[int],
    *,
    central_mesh_size_m: float,
    scope: str,
    robust_controls: RobustMinimaConfig | None = None,
    maximum_parallel_rows: int = 3,
    checkpoint_directory: str | Path | None = None,
) -> CalibrationEvaluation:
    """Evaluate one named hypothesis in a fresh process per dataset row."""

    selected_rows = tuple(int(item) for item in row_numbers)
    if not selected_rows or len(set(selected_rows)) != len(selected_rows):
        raise ValueError("row_numbers must be nonempty and unique")
    if any(item < 1 or item > dataset.row_count for item in selected_rows):
        raise ValueError("row_numbers contains an out-of-range row")
    if maximum_parallel_rows <= 0:
        raise ValueError("maximum_parallel_rows must be positive")
    geometry_config = case.geometry.geometry_config(case.voltage)
    config = locally_refined_real_scale_forward_config(
        central_mesh_size_m=central_mesh_size_m,
        geometry=geometry_config,
    )
    variant = ReferenceValidationVariant(
        name=case.name,
        displacement_mode="absolute",
        electrode_permutation=case.electrode_mapping,
        polarity_name=case.voltage.name,
    )
    controls = robust_controls or RobustMinimaConfig()
    checkpoint_path = _evaluation_checkpoint_path(
        checkpoint_directory,
        case,
        scope,
        central_mesh_size_m,
        selected_rows,
        controls,
    )
    if checkpoint_path is not None and checkpoint_path.exists():
        cached = pickle.loads(checkpoint_path.read_bytes())
        if not isinstance(cached, CalibrationEvaluation):
            raise RuntimeError(f"invalid calibration checkpoint: {checkpoint_path}")
        return cached
    prepared = []
    for row_number in selected_rows:
        index = row_number - 1
        solver_displacements, reference, row_config = prepare_reference_row_inputs(
            dataset.raw_displacements_m[index],
            dataset.raw_minima_absolute_m[index],
            config,
            variant,
        )
        prepared.append((row_number, solver_displacements, reference, row_config))
    with ThreadPoolExecutor(
        max_workers=min(maximum_parallel_rows, len(prepared))
    ) as executor:
        outcomes = tuple(
            executor.map(
                lambda item: _run_minima_worker(item[1], item[3], controls),
                prepared,
            )
        )
    rows = []
    for item, outcome in zip(prepared, outcomes, strict=True):
        row_number, _, reference, _ = item
        rows.append(
            _calibration_row_from_outcome(
                case,
                scope,
                row_number,
                reference,
                outcome,
            )
        )
    evaluation = CalibrationEvaluation(
        case=case,
        scope=scope,
        central_mesh_size_m=central_mesh_size_m,
        rows=tuple(rows),
    )
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_path.with_suffix(".tmp")
        temporary.write_bytes(pickle.dumps(evaluation))
        temporary.replace(checkpoint_path)
    return evaluation


def refit_evaluation_output(
    evaluation: CalibrationEvaluation,
    *,
    name_suffix: str = "calibrated-output",
) -> CalibrationEvaluation:
    """Fit and apply a bounded output transform without rerunning the FEM."""

    completed = [row for row in evaluation.rows if row.completed]
    if not completed:
        return evaluation
    transform = fit_output_transform(
        [row.raw_computed_positions_m for row in completed],
        [row.reference_positions_m for row in completed],
    )
    case = replace(
        evaluation.case,
        name=f"{evaluation.case.name}__{name_suffix}",
        family=f"{evaluation.case.family}+output-transform",
        output_transform=transform,
    )
    return _retransform_evaluation(evaluation, case)


def run_local_refinement_diagnostics(
    dataset: ReferenceDataset,
    *,
    row_numbers: Sequence[int] = (1, 2, 3),
    central_mesh_sizes_m: Sequence[float] = DEFAULT_REFINEMENT_SIZES_M,
    maximum_estimated_central_triangles: int = 100_000,
) -> tuple[tuple[LocalRefinementRecord, ...], float, float]:
    """Run feasible local meshes and retain explicit preflight-skipped cases.

    Returns records, the practical mesh size, and fractional mean-error
    improvement relative to the 500 micrometre local baseline.
    """

    if maximum_estimated_central_triangles <= 0:
        raise ValueError("maximum_estimated_central_triangles must be positive")
    case = CalibrationCase(
        name="local-refinement-perm1324",
        family="local-refinement",
        geometry=default_geometry_variant(),
        voltage=voltage_model_variants()[0],
    )
    records: list[LocalRefinementRecord] = []
    previous_by_row: dict[int, NDArray[np.float64]] = {}
    successful_evaluations: list[CalibrationEvaluation] = []
    for mesh_size in central_mesh_sizes_m:
        mesh_size = float(mesh_size)
        estimate = estimate_central_triangle_count(8.0e-3, mesh_size)
        if estimate > maximum_estimated_central_triangles:
            reason = (
                f"preflight estimate {estimate:,} central triangles exceeds "
                f"the {maximum_estimated_central_triangles:,} direct-solve limit"
            )
            records.extend(
                LocalRefinementRecord(
                    central_mesh_size_m=mesh_size,
                    estimated_central_triangles=estimate,
                    row_number=int(row_number),
                    status="skipped-impractical",
                    skip_or_failure_reason=reason,
                )
                for row_number in row_numbers
            )
            continue
        evaluation = evaluate_calibration_case(
            dataset,
            case,
            row_numbers,
            central_mesh_size_m=mesh_size,
            scope="local-refinement-rows1-3",
            maximum_parallel_rows=(1 if mesh_size < 500.0e-6 else 3),
        )
        successful_evaluations.append(evaluation)
        for row in evaluation.rows:
            branch_shift = float("nan")
            if row.completed and row.row_number in previous_by_row:
                branch_matches = match_minima_by_distance(
                    previous_by_row[row.row_number],
                    row.raw_computed_positions_m,
                )
                branch_shift = float(
                    np.max([item.distance_m for item in branch_matches])
                )
            if row.completed:
                previous_by_row[row.row_number] = row.raw_computed_positions_m.copy()
            errors = row.errors_m()
            records.append(
                LocalRefinementRecord(
                    central_mesh_size_m=mesh_size,
                    estimated_central_triangles=estimate,
                    row_number=row.row_number,
                    status=row.status,
                    node_count=row.node_count,
                    triangle_count=row.triangle_count,
                    runtime_seconds=row.runtime_seconds,
                    exactly_three_robust_minima=row.exactly_three_topology,
                    topology_candidate_count=row.topology_candidate_count,
                    selected_positions_m=(
                        row.raw_computed_positions_m.copy() if row.completed else None
                    ),
                    branch_shift_m=branch_shift,
                    mean_reference_error_m=(
                        float(np.mean(errors)) if errors.size else float("nan")
                    ),
                    maximum_reference_error_m=(
                        float(np.max(errors)) if errors.size else float("nan")
                    ),
                    rejected_candidates=row.rejected_candidates,
                    flagged_candidates=row.selected_interpolation_sensitive,
                    skip_or_failure_reason=(
                        "" if row.completed else f"{row.error_type}: {row.error_message}"
                    ),
                )
            )
    if not successful_evaluations:
        raise RuntimeError("all requested local refinements were preflight-skipped")
    ranked = rank_calibration_evaluations(successful_evaluations)
    best_mean = ranked[0].summary().mean_error_m
    practical_candidates = [
        item
        for item in successful_evaluations
        if np.isfinite(item.summary().mean_error_m)
        and item.summary().mean_error_m <= 1.02 * best_mean
        and item.summary().exactly_three_rows == len(row_numbers)
    ]
    practical = max(
        practical_candidates or [ranked[0]],
        key=lambda item: item.central_mesh_size_m,
    )
    baseline = min(
        successful_evaluations,
        key=lambda item: abs(item.central_mesh_size_m - 500.0e-6),
    )
    baseline_mean = baseline.summary().mean_error_m
    improvement = (
        (baseline_mean - practical.summary().mean_error_m) / baseline_mean
        if np.isfinite(baseline_mean) and baseline_mean > 0.0
        else float("nan")
    )
    return tuple(records), practical.central_mesh_size_m, float(improvement)


def resume_local_refinement_diagnostics(
    dataset: ReferenceDataset,
    *,
    checkpoint_directory: str | Path | None = None,
) -> tuple[tuple[LocalRefinementRecord, ...], float, float]:
    """Resume from the interrupted local sweep without discarding its evidence.

    The interrupted session printed aggregate 500 micrometre evidence, one
    completed 200 micrometre pilot, incomplete 200/100 micrometre attempts, and
    a manually terminated all-modes 50 micrometre attempt. This function
    re-runs only the complete 500 micrometre row set and the successful 200
    micrometre row-1 pilot with the robust-only worker so their row-level
    coordinates are serializable. Failed/timeout records are retained exactly
    as incomplete evidence and are not promoted into convergence claims.
    """

    case = CalibrationCase(
        name="local-refinement-perm1324",
        family="local-refinement",
        geometry=default_geometry_variant(),
        voltage=voltage_model_variants()[0],
    )
    records: list[LocalRefinementRecord] = []
    completed_by_size: dict[float, CalibrationEvaluation] = {}
    for mesh_size, rows in ((500.0e-6, (1, 2, 3)), (200.0e-6, (1,))):
        evaluation = evaluate_calibration_case(
            dataset,
            case,
            rows,
            central_mesh_size_m=mesh_size,
            scope="resumed-local-refinement",
            maximum_parallel_rows=(3 if mesh_size == 500.0e-6 else 1),
            checkpoint_directory=checkpoint_directory,
        )
        completed_by_size[mesh_size] = evaluation
        estimate = estimate_central_triangle_count(8.0e-3, mesh_size)
        for row in evaluation.rows:
            errors = row.errors_m()
            records.append(
                LocalRefinementRecord(
                    central_mesh_size_m=mesh_size,
                    estimated_central_triangles=estimate,
                    row_number=row.row_number,
                    status=row.status,
                    node_count=row.node_count,
                    triangle_count=row.triangle_count,
                    runtime_seconds=row.runtime_seconds,
                    exactly_three_robust_minima=row.exactly_three_topology,
                    topology_candidate_count=row.topology_candidate_count,
                    selected_positions_m=(
                        row.raw_computed_positions_m.copy() if row.completed else None
                    ),
                    mean_reference_error_m=(
                        float(np.mean(errors)) if errors.size else float("nan")
                    ),
                    maximum_reference_error_m=(
                        float(np.max(errors)) if errors.size else float("nan")
                    ),
                    rejected_candidates=row.rejected_candidates,
                    flagged_candidates=row.selected_interpolation_sensitive,
                    skip_or_failure_reason=(
                        "reproduced after interruption"
                        if row.completed
                        else f"{row.error_type}: {row.error_message}"
                    ),
                )
            )
    coarse_row_1 = next(
        item
        for item in completed_by_size[500.0e-6].rows
        if item.row_number == 1 and item.completed
    )
    fine_row_1 = completed_by_size[200.0e-6].rows[0]
    if fine_row_1.completed:
        branch_matches = match_minima_by_distance(
            coarse_row_1.raw_computed_positions_m,
            fine_row_1.raw_computed_positions_m,
        )
        branch_shift = float(max(item.distance_m for item in branch_matches))
        fine_index = next(
            index
            for index, item in enumerate(records)
            if item.central_mesh_size_m == 200.0e-6 and item.row_number == 1
        )
        records[fine_index] = replace(records[fine_index], branch_shift_m=branch_shift)

    records.extend(
        LocalRefinementRecord(
            central_mesh_size_m=200.0e-6,
            estimated_central_triangles=estimate_central_triangle_count(
                8.0e-3, 200.0e-6
            ),
            row_number=row_number,
            status="prior-run-timeout",
            runtime_seconds=300.0,
            skip_or_failure_reason=(
                "interrupted-session robust-only worker did not return within "
                "the then-current 300 s cap; no mesh counts or coordinates were retained"
            ),
        )
        for row_number in (2, 3)
    )
    records.extend(
        LocalRefinementRecord(
            central_mesh_size_m=100.0e-6,
            estimated_central_triangles=estimate_central_triangle_count(
                8.0e-3, 100.0e-6
            ),
            row_number=row_number,
            status="prior-run-failure-or-timeout",
            skip_or_failure_reason=(
                "interrupted concurrent robust-only attempt completed 0/3 rows; "
                "the process-level failure detail was not persisted"
            ),
        )
        for row_number in (1, 2, 3)
    )
    records.append(
        LocalRefinementRecord(
            central_mesh_size_m=50.0e-6,
            estimated_central_triangles=estimate_central_triangle_count(
                8.0e-3, 50.0e-6
            ),
            row_number=1,
            status="prior-run-manually-terminated",
            skip_or_failure_reason=(
                "obsolete all-modes pilot exceeded 600 s and was terminated; "
                "it returned no mesh counts, minima, or error metrics"
            ),
        )
    )
    records.extend(
        LocalRefinementRecord(
            central_mesh_size_m=50.0e-6,
            estimated_central_triangles=estimate_central_triangle_count(
                8.0e-3, 50.0e-6
            ),
            row_number=row_number,
            status="skipped-after-impractical-pilot",
            skip_or_failure_reason="row 1 pilot exceeded 600 s; rows 2--3 were not launched",
        )
        for row_number in (2, 3)
    )
    for mesh_size in DEFAULT_REFINEMENT_SIZES_M[4:]:
        estimate = estimate_central_triangle_count(8.0e-3, mesh_size)
        records.extend(
            LocalRefinementRecord(
                central_mesh_size_m=mesh_size,
                estimated_central_triangles=estimate,
                row_number=row_number,
                status="skipped-impractical",
                skip_or_failure_reason=(
                    f"preflight estimate {estimate:,} central triangles plus the "
                    "50 um runtime evidence made this direct solve impractical"
                ),
            )
            for row_number in (1, 2, 3)
        )
    coarse_mean = completed_by_size[500.0e-6].summary().mean_error_m
    fine_row_mean = (
        float(np.mean(fine_row_1.errors_m())) if fine_row_1.completed else float("nan")
    )
    coarse_row_mean = float(np.mean(coarse_row_1.errors_m()))
    row_1_improvement = (
        (coarse_row_mean - fine_row_mean) / coarse_row_mean
        if np.isfinite(fine_row_mean) and coarse_row_mean > 0.0
        else float("nan")
    )
    if not np.isfinite(coarse_mean):
        raise RuntimeError("resumed 500 um row set did not complete")
    return tuple(records), 500.0e-6, float(row_1_improvement)


def fit_refined_voltage_model(
    dataset: ReferenceDataset,
    *,
    row_numbers: Sequence[int] = (1, 2, 3),
    central_mesh_size_m: float = 500.0e-6,
    electrode_mapping: tuple[int, int, int, int] = BEST_KNOWN_MAPPING,
) -> VoltageModel:
    """Fit a normalized four-electrode vector from refined one-hot fields."""

    geometry = default_geometry_variant().geometry_config(
        VoltageModel("basis-placeholder", (1.0, 1.0, 1.0, 1.0))
    )
    config = locally_refined_real_scale_forward_config(
        central_mesh_size_m=central_mesh_size_m,
        geometry=geometry,
    )
    variant = ReferenceValidationVariant(
        name="refined-basis-fit",
        electrode_permutation=electrode_mapping,
        polarity_name="one-hot-basis",
    )
    usable_fields = []
    for row_number in row_numbers:
        index = int(row_number) - 1
        solver_displacements, reference, row_config = prepare_reference_row_inputs(
            dataset.raw_displacements_m[index],
            dataset.raw_minima_absolute_m[index],
            config,
            variant,
        )
        outcome = _run_basis_worker(
            solver_displacements,
            row_config,
            reference,
        )
        if bool(outcome.get("ok")):
            usable_fields.append(np.asarray(outcome["basis_fields_v_per_m"]))
    if not usable_fields:
        raise RuntimeError("all refined one-electrode basis diagnostics failed")
    potentials, _, _ = fit_basis_potentials(usable_fields)
    normalized, outer = normalize_voltage_vector(potentials)
    return VoltageModel(
        "basis-fit-milestone-9-refined",
        normalized,
        outer,
        origin=(
            "one-electrode-basis-fit-rows-"
            + "-".join(map(str, row_numbers))
            + f"-central-h-{central_mesh_size_m:.9g}m"
        ),
    )


def run_milestone_9_study(
    dataset: ReferenceDataset,
    *,
    promotion_row_count: int = 20,
    maximum_estimated_central_triangles: int = 100_000,
    resume_interrupted_local: bool = False,
    checkpoint_directory: str | Path | None = None,
) -> Milestone9Study:
    """Run refinement, coordinate-descent calibration, and promotion stages."""

    started = time.perf_counter()
    if resume_interrupted_local:
        local, chosen_h, improvement = resume_local_refinement_diagnostics(
            dataset,
            checkpoint_directory=checkpoint_directory,
        )
    else:
        local, chosen_h, improvement = run_local_refinement_diagnostics(
            dataset,
            maximum_estimated_central_triangles=maximum_estimated_central_triangles,
        )
    screening_h = 500.0e-6
    screening_rows = (1, 2, 3)
    all_positive = voltage_model_variants()[0]

    geometry_evaluations = []
    for geometry in generate_geometry_variants():
        case = CalibrationCase(
            name=f"geometry__{geometry.name}",
            family="geometry-calibration",
            geometry=geometry,
            voltage=all_positive,
        )
        raw = evaluate_calibration_case(
            dataset,
            case,
            screening_rows,
            central_mesh_size_m=screening_h,
            scope="geometry-screen-rows1-3",
            checkpoint_directory=checkpoint_directory,
        )
        geometry_evaluations.extend((raw, refit_evaluation_output(raw)))

    fitted_voltage = fit_refined_voltage_model(
        dataset,
        row_numbers=screening_rows,
        central_mesh_size_m=screening_h,
    )
    voltage_evaluations = []
    for voltage in (*voltage_model_variants(), fitted_voltage):
        case = CalibrationCase(
            name=f"voltage__{voltage.name}",
            family="voltage-calibration",
            geometry=default_geometry_variant(),
            voltage=voltage,
        )
        raw = evaluate_calibration_case(
            dataset,
            case,
            screening_rows,
            central_mesh_size_m=screening_h,
            scope="voltage-screen-rows1-3",
            checkpoint_directory=checkpoint_directory,
        )
        voltage_evaluations.extend((raw, refit_evaluation_output(raw)))

    raw_geometry_evaluations = tuple(geometry_evaluations[::2])
    raw_voltage_evaluations = tuple(voltage_evaluations[::2])
    best_geometry = rank_calibration_evaluations(raw_geometry_evaluations)[0].case.geometry
    best_voltage = rank_calibration_evaluations(raw_voltage_evaluations)[0].case.voltage
    mapping_evaluations = []
    for tail in itertools.permutations((2, 3, 4)):
        mapping = (1, *tail)
        case = CalibrationCase(
            name="mapping__" + "".join(map(str, mapping)),
            family="mapping-calibration",
            geometry=best_geometry,
            voltage=best_voltage,
            electrode_mapping=mapping,
        )
        raw = evaluate_calibration_case(
            dataset,
            case,
            screening_rows,
            central_mesh_size_m=screening_h,
            scope="mapping-screen-rows1-3",
            checkpoint_directory=checkpoint_directory,
        )
        mapping_evaluations.extend((raw, refit_evaluation_output(raw)))
    raw_mapping_evaluations = tuple(mapping_evaluations[::2])
    best_mapping = rank_calibration_evaluations(raw_mapping_evaluations)[0].case.electrode_mapping

    top_geometries = _unique_ranked_attribute(
        rank_calibration_evaluations(raw_geometry_evaluations),
        "geometry",
        2,
    )
    top_voltages = _unique_ranked_attribute(
        rank_calibration_evaluations(raw_voltage_evaluations),
        "voltage",
        2,
    )
    top_mappings = [
        best_mapping,
        BEST_KNOWN_MAPPING if best_mapping != BEST_KNOWN_MAPPING else (1, 2, 3, 4),
    ]
    combined_screen = []
    for geometry, voltage, mapping in itertools.product(
        top_geometries,
        top_voltages,
        top_mappings,
    ):
        name = (
            f"combined__{geometry.name}__{voltage.name}__perm"
            + "".join(map(str, mapping))
        )
        case = CalibrationCase(
            name=name,
            family="combined-calibration",
            geometry=geometry,
            voltage=voltage,
            electrode_mapping=mapping,
        )
        raw = evaluate_calibration_case(
            dataset,
            case,
            screening_rows,
            central_mesh_size_m=screening_h,
            scope="combined-screen-rows1-3",
            checkpoint_directory=checkpoint_directory,
        )
        combined_screen.extend((raw, refit_evaluation_output(raw)))

    raw_combined_screen = tuple(combined_screen[::2])
    physical_candidates: dict[
        tuple[GeometryVariant, VoltageModel, tuple[int, int, int, int]],
        CalibrationCase,
    ] = {}

    def add_physical_case(case: CalibrationCase) -> None:
        signature = (case.geometry, case.voltage, case.electrode_mapping)
        physical_candidates.setdefault(signature, replace(case, output_transform=OutputTransform()))

    baseline_case = CalibrationCase(
        name="combined__baseline-real-scale__all-positive__perm1324",
        family="combined-calibration-baseline",
        geometry=default_geometry_variant(),
        voltage=all_positive,
        electrode_mapping=BEST_KNOWN_MAPPING,
    )
    add_physical_case(baseline_case)
    add_physical_case(
        CalibrationCase(
            name=f"combined__{best_geometry.name}__all-positive__perm1324",
            family="combined-calibration",
            geometry=best_geometry,
            voltage=all_positive,
            electrode_mapping=BEST_KNOWN_MAPPING,
        )
    )
    add_physical_case(
        CalibrationCase(
            name=f"combined__real-scale-default__{best_voltage.name}__perm1324",
            family="combined-calibration",
            geometry=default_geometry_variant(),
            voltage=best_voltage,
            electrode_mapping=BEST_KNOWN_MAPPING,
        )
    )
    for item in rank_calibration_evaluations(raw_combined_screen)[:4]:
        add_physical_case(item.case)

    rows_1_10 = tuple(range(1, min(10, dataset.row_count) + 1))
    combined_rows_1_10_list = []
    for case in physical_candidates.values():
        raw = evaluate_calibration_case(
            dataset,
            case,
            rows_1_10,
            central_mesh_size_m=chosen_h,
            scope="combined-rows1-10",
            checkpoint_directory=checkpoint_directory,
        )
        combined_rows_1_10_list.extend((raw, refit_evaluation_output(raw)))
    combined_rows_1_10 = tuple(combined_rows_1_10_list)
    if promotion_row_count < 10 or promotion_row_count > dataset.row_count:
        raise ValueError("promotion_row_count must lie between 10 and dataset row count")
    promoted = []
    for item in rank_calibration_evaluations(combined_rows_1_10)[:3]:
        extra_rows = tuple(range(11, promotion_row_count + 1))
        if extra_rows:
            extra = evaluate_calibration_case(
                dataset,
                item.case,
                extra_rows,
                central_mesh_size_m=chosen_h,
                scope=f"promoted-rows1-{promotion_row_count}",
                checkpoint_directory=checkpoint_directory,
            )
            combined_rows = (*item.rows, *extra.rows)
        else:
            combined_rows = item.rows
        promotion_scope = f"promoted-rows1-{promotion_row_count}"
        promoted.append(
            CalibrationEvaluation(
                case=item.case,
                scope=promotion_scope,
                central_mesh_size_m=chosen_h,
                rows=tuple(replace(row, scope=promotion_scope) for row in combined_rows),
            )
        )
    return Milestone9Study(
        local_refinement=local,
        geometry_evaluations=tuple(geometry_evaluations),
        voltage_evaluations=tuple(voltage_evaluations),
        combined_evaluations=tuple(combined_screen) + combined_rows_1_10,
        promoted_evaluations=tuple(promoted),
        chosen_central_mesh_size_m=chosen_h,
        local_refinement_improvement_fraction=improvement,
        fitted_voltage=fitted_voltage,
        runtime_seconds=time.perf_counter() - started,
        resumed_from_interrupted_local_run=resume_interrupted_local,
    )


def _evaluation_checkpoint_path(
    checkpoint_directory: str | Path | None,
    case: CalibrationCase,
    scope: str,
    central_mesh_size_m: float,
    row_numbers: Sequence[int],
    controls: RobustMinimaConfig,
) -> Path | None:
    if checkpoint_directory is None:
        return None
    payload = pickle.dumps(
        (
            "milestone-9-checkpoint-v3-absolute-displacements",
            case,
            scope,
            float(central_mesh_size_m),
            tuple(row_numbers),
            controls,
        )
    )
    digest = hashlib.sha256(payload).hexdigest()[:20]
    return Path(checkpoint_directory) / f"evaluation_{digest}.pickle"


def _run_minima_worker(
    displacements_m: NDArray[np.float64],
    config: ForwardModelConfig,
    controls: RobustMinimaConfig,
) -> dict[str, object]:
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "rf_trap_forward._calibrated_worker"],
            input=pickle.dumps((displacements_m, config, controls)),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=180.0,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "error_type": "WorkerTimeout",
            "error_message": "fresh FEM process exceeded 180 seconds",
            "runtime_seconds": 180.0,
        }
    if completed.returncode != 0:
        return {
            "ok": False,
            "error_type": "WorkerProcessError",
            "error_message": completed.stderr.decode("utf-8", errors="replace").strip(),
        }
    try:
        outcome = pickle.loads(completed.stdout)
    except Exception as error:
        return {
            "ok": False,
            "error_type": "WorkerProtocolError",
            "error_message": str(error),
        }
    if not isinstance(outcome, dict) or "ok" not in outcome:
        return {
            "ok": False,
            "error_type": "WorkerProtocolError",
            "error_message": "worker returned an invalid object",
        }
    return outcome


def _run_basis_worker(
    displacements_m: NDArray[np.float64],
    config: ForwardModelConfig,
    target_positions_m: NDArray[np.float64],
) -> dict[str, object]:
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "rf_trap_forward._basis_fit_worker"],
            input=pickle.dumps((displacements_m, config, target_positions_m)),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=180.0,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "error_type": "BasisWorkerTimeout",
            "error_message": "one-hot basis process exceeded 180 seconds",
        }
    if completed.returncode != 0:
        return {
            "ok": False,
            "error_type": "BasisWorkerProcessError",
            "error_message": completed.stderr.decode("utf-8", errors="replace").strip(),
        }
    try:
        outcome = pickle.loads(completed.stdout)
    except Exception as error:
        return {
            "ok": False,
            "error_type": "BasisWorkerProtocolError",
            "error_message": str(error),
        }
    return outcome if isinstance(outcome, dict) else {"ok": False}


def _calibration_row_from_outcome(
    case: CalibrationCase,
    scope: str,
    row_number: int,
    reference_positions_m: NDArray[np.float64],
    outcome: dict[str, object],
) -> CalibrationRow:
    empty = np.empty((0, 2), dtype=float)
    common = {
        "hypothesis_name": case.name,
        "family": case.family,
        "scope": scope,
        "row_number": row_number,
        "reference_positions_m": reference_positions_m.copy(),
    }
    if not bool(outcome.get("ok")):
        return CalibrationRow(
            **common,
            status="worker-failed",
            raw_computed_positions_m=empty,
            computed_positions_m=empty,
            matches=(),
            topology_candidate_count=0,
            selected_interpolation_sensitive=0,
            rejected_candidates=0,
            total_candidates=0,
            node_count=0,
            triangle_count=0,
            relative_free_residual=float("nan"),
            runtime_seconds=float(outcome.get("runtime_seconds", 0.0)),
            error_type=str(outcome.get("error_type", "WorkerError")),
            error_message=str(outcome.get("error_message", "unknown error")),
        )
    modes = outcome.get("modes", {})
    if "robust" not in modes:
        error_type, error_message = outcome.get("mode_errors", {}).get(
            "robust",
            ("ModeMissing", "worker did not return robust mode"),
        )
        return CalibrationRow(
            **common,
            status="mode-failed",
            raw_computed_positions_m=empty,
            computed_positions_m=empty,
            matches=(),
            topology_candidate_count=0,
            selected_interpolation_sensitive=0,
            rejected_candidates=0,
            total_candidates=0,
            node_count=int(outcome["node_count"]),
            triangle_count=int(outcome["triangle_count"]),
            relative_free_residual=float(outcome["relative_free_residual"]),
            runtime_seconds=float(outcome["runtime_seconds"]),
            error_type=str(error_type),
            error_message=str(error_message),
        )
    result: MinimaModeResult = modes["robust"]
    raw = _positions(result.minima)
    computed = apply_output_transform(raw, case.output_transform)
    matches = (
        match_minima_by_distance(reference_positions_m, computed)
        if computed.shape == (3, 2)
        else ()
    )
    return CalibrationRow(
        **common,
        status="ok" if len(matches) == 3 else "insufficient-minima",
        raw_computed_positions_m=raw,
        computed_positions_m=computed,
        matches=matches,
        topology_candidate_count=result.accepted_candidates,
        selected_interpolation_sensitive=result.selected_interpolation_sensitive,
        rejected_candidates=result.rejected_candidates,
        total_candidates=len(result.candidates),
        node_count=int(outcome["node_count"]),
        triangle_count=int(outcome["triangle_count"]),
        relative_free_residual=float(outcome["relative_free_residual"]),
        runtime_seconds=float(outcome["runtime_seconds"]),
    )


def _retransform_evaluation(
    evaluation: CalibrationEvaluation,
    case: CalibrationCase,
) -> CalibrationEvaluation:
    rows = []
    for old in evaluation.rows:
        computed = (
            apply_output_transform(old.raw_computed_positions_m, case.output_transform)
            if old.raw_computed_positions_m.size
            else old.raw_computed_positions_m.copy()
        )
        matches = (
            match_minima_by_distance(old.reference_positions_m, computed)
            if computed.shape == (3, 2)
            else ()
        )
        rows.append(
            replace(
                old,
                hypothesis_name=case.name,
                family=case.family,
                computed_positions_m=computed,
                matches=matches,
                status="ok" if len(matches) == 3 else old.status,
            )
        )
    return CalibrationEvaluation(
        case=case,
        scope=evaluation.scope,
        central_mesh_size_m=evaluation.central_mesh_size_m,
        rows=tuple(rows),
    )


def _positions(minima: Sequence[LocalMinimum]) -> NDArray[np.float64]:
    if not minima:
        return np.empty((0, 2), dtype=float)
    return np.vstack([item.position_m for item in minima])


def _unique_ranked_attribute(
    evaluations: Sequence[CalibrationEvaluation],
    attribute: str,
    count: int,
) -> list[object]:
    selected = []
    names = set()
    for item in evaluations:
        value = getattr(item.case, attribute)
        name = value.name
        if name in names:
            continue
        selected.append(value)
        names.add(name)
        if len(selected) == count:
            break
    return selected


def write_milestone_9_outputs(
    study: Milestone9Study,
    output_directory: str | Path,
) -> Milestone9OutputPaths:
    """Write required CSVs, Markdown reports, and diagnostic plots."""

    output = Path(output_directory)
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    plot_paths = (
        plots / "local_refinement_error.png",
        plots / "local_refinement_cost.png",
        plots / "calibration_ranking.png",
        plots / "best_reference_vs_computed.png",
        plots / "best_per_row_error.png",
        plots / "best_error_vectors.png",
    )
    paths = Milestone9OutputPaths(
        local_csv=output / "local_refinement_diagnostics.csv",
        local_report_markdown=output / "local_refinement_report.md",
        geometry_csv=output / "geometry_calibration_summary.csv",
        voltage_csv=output / "voltage_model_summary.csv",
        combined_csv=output / "combined_calibration_summary.csv",
        per_row_csv=output / "per_row_best_calibrated.csv",
        per_minimum_csv=output / "per_minimum_best_calibrated.csv",
        report_markdown=output / "milestone_9_report.md",
        plot_paths=plot_paths,
    )
    _write_csv(paths.local_csv, [_local_record(item) for item in study.local_refinement])
    _write_csv(
        paths.geometry_csv,
        [_summary_record(item.summary()) for item in study.geometry_evaluations],
    )
    _write_csv(
        paths.voltage_csv,
        [_summary_record(item.summary()) for item in study.voltage_evaluations],
    )
    _write_csv(
        paths.combined_csv,
        [
            _summary_record(item.summary())
            for item in (*study.combined_evaluations, *study.promoted_evaluations)
        ],
    )
    best = rank_calibration_evaluations(study.promoted_evaluations)[0]
    _write_csv(paths.per_row_csv, [_row_record(row) for row in best.rows])
    _write_csv(
        paths.per_minimum_csv,
        [
            _minimum_record(best.case, row, match)
            for row in best.rows
            for match in row.matches
        ],
    )
    paths.local_report_markdown.write_text(
        _local_refinement_markdown(study),
        encoding="utf-8",
    )
    paths.report_markdown.write_text(_milestone_markdown(study), encoding="utf-8")
    _plot_local_refinement_error(study, plot_paths[0])
    _plot_local_refinement_cost(study, plot_paths[1])
    _plot_calibration_ranking(study, plot_paths[2])
    _plot_reference_vs_computed(best, plot_paths[3])
    _plot_per_row_error(best, plot_paths[4])
    _plot_error_vectors(best, plot_paths[5])
    return paths


def _local_record(item: LocalRefinementRecord) -> dict[str, object]:
    positions = item.selected_positions_m
    record: dict[str, object] = {
        "central_mesh_size_m": item.central_mesh_size_m,
        "central_mesh_size_um": 1.0e6 * item.central_mesh_size_m,
        "central_region_radius_m": 8.0e-3,
        "central_region_radius_mm": 8.0,
        "outer_mesh_size_m": 2.0e-3,
        "outer_mesh_size_mm": 2.0,
        "electrode_boundary_mesh_size_m": 0.50e-3,
        "electrode_boundary_mesh_size_mm": 0.50,
        "estimated_central_triangles": item.estimated_central_triangles,
        "row_number": item.row_number,
        "status": item.status,
        "node_count": item.node_count,
        "triangle_count": item.triangle_count,
        "runtime_seconds": item.runtime_seconds,
        "exactly_three_robust_minima": item.exactly_three_robust_minima,
        "topology_candidate_count": item.topology_candidate_count,
        "branch_shift_m": item.branch_shift_m,
        "branch_shift_um": 1.0e6 * item.branch_shift_m,
        "mean_reference_error_m": item.mean_reference_error_m,
        "mean_reference_error_mm": 1.0e3 * item.mean_reference_error_m,
        "maximum_reference_error_m": item.maximum_reference_error_m,
        "maximum_reference_error_mm": 1.0e3 * item.maximum_reference_error_m,
        "rejected_candidates": item.rejected_candidates,
        "flagged_candidates": item.flagged_candidates,
        "skip_or_failure_reason": item.skip_or_failure_reason,
    }
    for index in range(3):
        point = positions[index] if positions is not None and len(positions) > index else None
        record[f"minimum_{index + 1}_x_m"] = "" if point is None else point[0]
        record[f"minimum_{index + 1}_y_m"] = "" if point is None else point[1]
        record[f"minimum_{index + 1}_x_mm"] = "" if point is None else 1.0e3 * point[0]
        record[f"minimum_{index + 1}_y_mm"] = "" if point is None else 1.0e3 * point[1]
    return record


def _summary_record(item: CalibrationSummary) -> dict[str, object]:
    transform = item.output_transform
    return {
        "hypothesis": item.hypothesis_name,
        "family": item.family,
        "scope": item.scope,
        "geometry_name": item.geometry_name,
        "voltage_name": item.voltage_name,
        "electrode_mapping": "-".join(map(str, item.electrode_mapping)),
        "electrode_center_radius_m": item.electrode_center_radius_m,
        "electrode_center_radius_mm": 1.0e3 * item.electrode_center_radius_m,
        "electrode_radius_m": item.electrode_radius_m,
        "electrode_radius_mm": 1.0e3 * item.electrode_radius_m,
        "inner_surface_clearance_m": (
            item.electrode_center_radius_m - item.electrode_radius_m
        ),
        "inner_surface_clearance_mm": 1.0e3
        * (item.electrode_center_radius_m - item.electrode_radius_m),
        "outer_boundary_radius_m": item.outer_boundary_radius_m,
        "outer_boundary_radius_mm": 1.0e3 * item.outer_boundary_radius_m,
        "electrode_potential_1_v": item.electrode_potentials_v[0],
        "electrode_potential_2_v": item.electrode_potentials_v[1],
        "electrode_potential_3_v": item.electrode_potentials_v[2],
        "electrode_potential_4_v": item.electrode_potentials_v[3],
        "outer_potential_v": item.outer_potential_v,
        "global_output_scale": transform.global_scale,
        "rotation_deg": transform.rotation_deg,
        "anisotropy_ratio": transform.anisotropy_ratio,
        "x_output_scale": transform.x_scale,
        "y_output_scale": transform.y_scale,
        "central_mesh_size_m": item.central_mesh_size_m,
        "central_mesh_size_um": 1.0e6 * item.central_mesh_size_m,
        "selected_rows": item.selected_rows,
        "completed_rows": item.completed_rows,
        "exactly_three_rows": item.exactly_three_rows,
        "matched_minima": item.matched_minima,
        "mean_error_m": item.mean_error_m,
        "mean_error_mm": 1.0e3 * item.mean_error_m,
        "median_error_m": item.median_error_m,
        "median_error_mm": 1.0e3 * item.median_error_m,
        "maximum_error_m": item.maximum_error_m,
        "maximum_error_mm": 1.0e3 * item.maximum_error_m,
        "percentile_95_error_m": item.percentile_95_error_m,
        "percentile_95_error_mm": 1.0e3 * item.percentile_95_error_m,
        "selected_interpolation_sensitive": item.selected_interpolation_sensitive,
        "rejected_candidates": item.rejected_candidates,
        "runtime_seconds": item.runtime_seconds,
        "validation_gate_passed": item.validation_gate_passed,
    }


def _row_record(item: CalibrationRow) -> dict[str, object]:
    errors = item.errors_m()
    return {
        "hypothesis": item.hypothesis_name,
        "family": item.family,
        "scope": item.scope,
        "row_number": item.row_number,
        "status": item.status,
        "completed": item.completed,
        "exactly_three_topology": item.exactly_three_topology,
        "topology_candidate_count": item.topology_candidate_count,
        "mean_error_m": float(np.mean(errors)) if errors.size else float("nan"),
        "mean_error_mm": 1.0e3 * float(np.mean(errors)) if errors.size else float("nan"),
        "maximum_error_m": float(np.max(errors)) if errors.size else float("nan"),
        "maximum_error_mm": 1.0e3 * float(np.max(errors)) if errors.size else float("nan"),
        "node_count": item.node_count,
        "triangle_count": item.triangle_count,
        "relative_free_residual": item.relative_free_residual,
        "runtime_seconds": item.runtime_seconds,
        "selected_interpolation_sensitive": item.selected_interpolation_sensitive,
        "rejected_candidates": item.rejected_candidates,
        "total_candidates": item.total_candidates,
        "error_type": item.error_type,
        "error_message": item.error_message,
    }


def _minimum_record(
    case: CalibrationCase,
    row: CalibrationRow,
    match: MinimumMatch,
) -> dict[str, object]:
    return {
        "hypothesis": case.name,
        "scope": row.scope,
        "row_number": row.row_number,
        "reference_index": match.reference_index,
        "computed_index": match.computed_index,
        "reference_x_m": match.reference_position_m[0],
        "reference_y_m": match.reference_position_m[1],
        "computed_x_m": match.computed_position_m[0],
        "computed_y_m": match.computed_position_m[1],
        "delta_x_m": match.delta_m[0],
        "delta_y_m": match.delta_m[1],
        "distance_m": match.distance_m,
        "reference_x_mm": 1.0e3 * match.reference_position_m[0],
        "reference_y_mm": 1.0e3 * match.reference_position_m[1],
        "computed_x_mm": 1.0e3 * match.computed_position_m[0],
        "computed_y_mm": 1.0e3 * match.computed_position_m[1],
        "delta_x_mm": 1.0e3 * match.delta_m[0],
        "delta_y_mm": 1.0e3 * match.delta_m[1],
        "distance_mm": 1.0e3 * match.distance_m,
    }


def _write_csv(path: Path, records: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError(f"refusing to write empty table: {path}")
    fieldnames = list(records[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _local_mesh_summaries(
    records: Sequence[LocalRefinementRecord],
) -> list[tuple[float, int, int, float, float, float]]:
    summaries = []
    for mesh_size in sorted({item.central_mesh_size_m for item in records}, reverse=True):
        rows = [item for item in records if item.central_mesh_size_m == mesh_size]
        successful = [item for item in rows if item.status == "ok"]
        errors = [item.mean_reference_error_m for item in successful]
        summaries.append(
            (
                mesh_size,
                len(successful),
                sum(item.exactly_three_robust_minima for item in successful),
                float(np.mean(errors)) if errors else float("nan"),
                max((item.branch_shift_m for item in successful), default=float("nan")),
                sum(item.runtime_seconds for item in successful),
            )
        )
    return summaries


def _local_refinement_markdown(study: Milestone9Study) -> str:
    lines = [
        "# Milestone 9 local central-refinement report",
        "",
        "The outer domain remains coarse at 2 mm, electrode boundaries use 0.5 mm, and only the 8 mm-radius central disk is refined. The FEM physics and robust-minima rules are unchanged.",
        "",
        (
            "This report resumes an interrupted run. Completed configurations were "
            "reproduced with the robust-only worker so row-level coordinates could be "
            "serialized; prior timeout/failure/termination records were preserved and "
            "were not counted as successful convergence evidence."
            if study.resumed_from_interrupted_local_run
            else "This report was produced in one uninterrupted selector run."
        ),
        "",
        "| central h (um) | completed | exactly three | mean reference error (mm) | max branch shift (um) | runtime (s) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for mesh, completed, exact, mean, shift, runtime in _local_mesh_summaries(
        study.local_refinement
    ):
        lines.append(
            f"| {1.0e6 * mesh:.6g} | {completed}/3 | {exact}/3 | {_fmt_mm(mean)} | {_fmt_um(shift)} | {runtime:.3f} |"
        )
    skipped = [
        item
        for item in study.local_refinement
        if item.status.startswith("skipped") or item.status.startswith("prior-run")
    ]
    lines.extend(
        (
            "",
            f"Chosen practical central h: **{1.0e6 * study.chosen_central_mesh_size_m:.6g} um**. It is the only tested local mesh with completed exactly-three evidence for all rows 1--3; the 200 um result is row 1 only.",
            (
                "Row-1 mean-error change from 500 to 200 um: "
                f"**{abs(100.0 * study.local_refinement_improvement_fraction):.3f}% "
                f"{'improvement' if study.local_refinement_improvement_fraction >= 0.0 else 'worsening'}**. "
                "This is not a three-row convergence estimate because rows 2--3 timed out."
            ),
            "",
            "Meshes below the direct-solve preflight limit were run. Finer requests were preserved as skipped rows, not silently omitted.",
        )
    )
    if skipped:
        for mesh in sorted({item.central_mesh_size_m for item in skipped}, reverse=True):
            sample = next(item for item in skipped if item.central_mesh_size_m == mesh)
            lines.append(f"- {1.0e6 * mesh:.6g} um: {sample.skip_or_failure_reason}.")
    lines.extend(
        (
            "",
            "A 10% reduction with complete three-row topology was the promotion criterion for resolution alone. "
            + (
                "That criterion was met."
                if study.local_refinement_improvement_fraction >= 0.10
                else "That criterion was not met; central resolution alone is insufficient."
            ),
            "",
        )
    )
    return "\n".join(lines)


def _milestone_markdown(study: Milestone9Study) -> str:
    best_geometry = rank_calibration_evaluations(study.geometry_evaluations[::2])[0]
    best_voltage = rank_calibration_evaluations(study.voltage_evaluations[::2])[0]
    rows10 = [
        item
        for item in study.combined_evaluations
        if item.scope == "combined-rows1-10"
    ]
    best10 = rank_calibration_evaluations(rows10)[0]
    best_promoted = rank_calibration_evaluations(study.promoted_evaluations)[0]
    summaries = [
        ("Best raw geometry screen", best_geometry.summary()),
        ("Best raw voltage screen", best_voltage.summary()),
        ("Best combined rows 1--10", best10.summary()),
        ("Best promoted", best_promoted.summary()),
    ]
    promoted_count = best_promoted.summary().selected_rows
    gate = best_promoted.summary().validation_gate_passed
    lines = [
        "# Milestone 9 targeted refinement and calibration report",
        "",
        "## Scope and invariants",
        "",
        "No ML or synthetic dataset generation was performed. Local meshing, geometry, voltage, electrode mapping, and output transforms are explicit named diagnostics. The default all-positive real-scale model was not overwritten.",
        "",
        "## Best results",
        "",
        "| stage | hypothesis | rows | completed | exactly three | mean (mm) | median (mm) | max (mm) | p95 (mm) | gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for label, summary in summaries:
        lines.append(
            f"| {label} | `{summary.hypothesis_name}` | {summary.selected_rows} | {summary.completed_rows} | {summary.exactly_three_rows} | {_fmt_mm(summary.mean_error_m)} | {_fmt_mm(summary.median_error_m)} | {_fmt_mm(summary.maximum_error_m)} | {_fmt_mm(summary.percentile_95_error_m)} | {summary.validation_gate_passed} |"
        )
    promoted = best_promoted.summary()
    lines.extend(
        (
            "",
            "## Best calibrated hypothesis",
            "",
            f"- Geometry: `{promoted.geometry_name}`; center radius {1.0e3 * promoted.electrode_center_radius_m:.6g} mm, electrode radius {1.0e3 * promoted.electrode_radius_m:.6g} mm, outer radius {1.0e3 * promoted.outer_boundary_radius_m:.6g} mm.",
            f"- Electrode mapping: FEM E1--E4 <- source {','.join(map(str, promoted.electrode_mapping))}.",
            f"- Voltage model: `{promoted.voltage_name}`; electrodes {promoted.electrode_potentials_v}, outer {promoted.outer_potential_v:.9g} V.",
            f"- Diagnostic output transform: scale {promoted.output_transform.global_scale:.9g}, rotation {promoted.output_transform.rotation_deg:.6g} deg, anisotropy ratio {promoted.output_transform.anisotropy_ratio:.9g}.",
            f"- Central mesh h: {1.0e6 * promoted.central_mesh_size_m:.6g} um.",
            "",
            "## Interpretation",
            "",
            "The completed row-1 500-to-200 um comparison "
            f"{'improved' if study.local_refinement_improvement_fraction >= 0.0 else 'worsened'} mean error by "
            f"{abs(100.0 * study.local_refinement_improvement_fraction):.3f}%; the 200 um rows 2--3 attempts timed out. "
            + (
                "This is material by the specified 10% criterion."
                if study.local_refinement_improvement_fraction >= 0.10
                else "This is below the specified 10% material-improvement criterion."
            ),
            f"The best rows 1--10 calibrated result has mean {_fmt_mm(best10.summary().mean_error_m)} mm and max {_fmt_mm(best10.summary().maximum_error_m)} mm.",
            f"The corresponding untransformed real-scale baseline on rows 1--10 has mean {_fmt_mm(_find_rows10_baseline(study).summary().mean_error_m)} mm and max {_fmt_mm(_find_rows10_baseline(study).summary().maximum_error_m)} mm.",
            f"The promoted rows 1--{promoted_count} result has mean {_fmt_mm(promoted.mean_error_m)} mm and max {_fmt_mm(promoted.maximum_error_m)} mm, with exactly-three topology in {promoted.exactly_three_rows}/{promoted.selected_rows} rows.",
            "",
            "The validation gate " + ("passes." if gate else "fails."),
            "",
            (
                "The current calibrated four-electrode model is consistent enough to consider downstream generation only after independent confirmation."
                if gate
                else "Because mesh, geometry, coordinate, and static-voltage calibration do not meet the gate, the leading missing assumption is the physical electrode/drive model: Data.txt is associated with an octupole, while this solver still represents four circular electrode holes with one scalar Dirichlet potential per electrode. The full rod count, RF phase/amplitude grouping, or three-dimensional/end-effect physics may be necessary. This is a diagnostic conclusion, not a silent model change."
            ),
            "",
            "Synthetic dataset generation is **not safe** unless the gate is passed and Data.txt agreement is independently confirmed.",
            "",
            "## Runtime and promotion",
            "",
            f"Current resumed orchestration wall time: {study.runtime_seconds:.3f} s. Checkpointed solves from earlier passes were reused, and the interrupted-run wall time is not included. The best three hypotheses were promoted to rows 1--{promoted_count}; rows 1--50 were not used because repeating local-mesh direct solves for every calibrated case was not proportionate after the rows 1--10 gate failure.",
            "",
        )
    )
    return "\n".join(lines)


def _find_rows10_baseline(study: Milestone9Study) -> CalibrationEvaluation:
    for item in study.combined_evaluations:
        if (
            item.scope == "combined-rows1-10"
            and item.case.name == "combined__baseline-real-scale__all-positive__perm1324"
        ):
            return item
    raise RuntimeError("rows 1--10 real-scale baseline is missing")


def _fmt_mm(value_m: float) -> str:
    return "nan" if not np.isfinite(value_m) else f"{1.0e3 * value_m:.6g}"


def _fmt_um(value_m: float) -> str:
    return "nan" if not np.isfinite(value_m) else f"{1.0e6 * value_m:.6g}"


def _save_figure(figure: Figure, path: Path) -> None:
    figure.tight_layout()
    FigureCanvasAgg(figure).print_png(path)


def _plot_local_refinement_error(study: Milestone9Study, path: Path) -> None:
    successful = [item for item in study.local_refinement if item.status == "ok"]
    figure = Figure(figsize=(7.4, 4.7))
    axis = figure.subplots()
    for row_number in sorted({item.row_number for item in successful}):
        row_records = sorted(
            (item for item in successful if item.row_number == row_number),
            key=lambda item: item.central_mesh_size_m,
            reverse=True,
        )
        axis.plot(
            [1.0e6 * item.central_mesh_size_m for item in row_records],
            [1.0e3 * item.mean_reference_error_m for item in row_records],
            "o-",
            label=f"Data.txt row {row_number}",
        )
    axis.axhline(0.25, color="#DC2626", linestyle="--", label="mean-error gate")
    axis.set_xscale("log")
    axis.invert_xaxis()
    axis.set_xlabel("Central mesh size (um; finer to the right)")
    axis.set_ylabel("Per-row mean assignment error (mm)")
    axis.set_title("Targeted central refinement")
    axis.grid(alpha=0.25)
    axis.legend()
    _save_figure(figure, path)


def _plot_local_refinement_cost(study: Milestone9Study, path: Path) -> None:
    successful = [item for item in study.local_refinement if item.status == "ok"]
    figure = Figure(figsize=(7.4, 4.7))
    axis = figure.subplots()
    axis.scatter(
        [item.triangle_count for item in successful],
        [item.runtime_seconds for item in successful],
        c=[1.0e6 * item.central_mesh_size_m for item in successful],
        cmap="viridis_r",
    )
    axis.set_xlabel("Triangles")
    axis.set_ylabel("Fresh-process runtime (s)")
    axis.set_title("Local-refinement computational cost")
    axis.grid(alpha=0.25)
    _save_figure(figure, path)


def _plot_calibration_ranking(study: Milestone9Study, path: Path) -> None:
    evaluations = rank_calibration_evaluations(
        (*study.geometry_evaluations, *study.voltage_evaluations, *study.combined_evaluations)
    )[:12]
    labels = [item.case.name[:42] for item in evaluations][::-1]
    means = [1.0e3 * item.summary().mean_error_m for item in evaluations][::-1]
    figure = Figure(figsize=(9.2, 6.0))
    axis = figure.subplots()
    axis.barh(labels, means, color="#2563EB")
    axis.axvline(0.25, color="#DC2626", linestyle="--")
    axis.set_xlabel("Mean assignment error (mm)")
    axis.set_title("Best screened calibration hypotheses")
    axis.grid(axis="x", alpha=0.25)
    _save_figure(figure, path)


def _plot_reference_vs_computed(evaluation: CalibrationEvaluation, path: Path) -> None:
    figure = Figure(figsize=(6.4, 6.2))
    axis = figure.subplots()
    for row in evaluation.rows:
        if not row.completed:
            continue
        reference = 1.0e3 * row.reference_positions_m
        computed = 1.0e3 * row.computed_positions_m
        axis.scatter(reference[:, 0], reference[:, 1], marker="x", color="#111827", s=28)
        axis.scatter(computed[:, 0], computed[:, 1], facecolors="none", edgecolors="#2563EB", s=28)
    axis.scatter([], [], marker="x", color="#111827", label="Data.txt")
    axis.scatter([], [], facecolors="none", edgecolors="#2563EB", label="calibrated FEM")
    axis.set_xlabel("x (mm)")
    axis.set_ylabel("y (mm)")
    axis.set_aspect("equal")
    axis.set_title("Best promoted hypothesis: reference versus FEM")
    axis.grid(alpha=0.2)
    axis.legend()
    _save_figure(figure, path)


def _plot_per_row_error(evaluation: CalibrationEvaluation, path: Path) -> None:
    rows = [row for row in evaluation.rows if row.completed]
    figure = Figure(figsize=(8.0, 4.7))
    axis = figure.subplots()
    axis.bar(
        [row.row_number for row in rows],
        [1.0e3 * float(np.mean(row.errors_m())) for row in rows],
        color="#0F766E",
    )
    axis.axhline(0.25, color="#DC2626", linestyle="--", label="mean gate")
    axis.set_xlabel("Data.txt row")
    axis.set_ylabel("Mean assignment error (mm)")
    axis.set_title("Best promoted hypothesis per-row errors")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    _save_figure(figure, path)


def _plot_error_vectors(evaluation: CalibrationEvaluation, path: Path) -> None:
    figure = Figure(figsize=(6.4, 6.2))
    axis = figure.subplots()
    for row in evaluation.rows:
        for match in row.matches:
            reference = 1.0e3 * match.reference_position_m
            delta = 1.0e3 * match.delta_m
            axis.arrow(
                reference[0],
                reference[1],
                delta[0],
                delta[1],
                width=0.006,
                head_width=0.09,
                color="#7C3AED",
                alpha=0.55,
                length_includes_head=True,
            )
    axis.set_xlabel("reference x (mm)")
    axis.set_ylabel("reference y (mm)")
    axis.set_aspect("equal")
    axis.set_title("Best promoted hypothesis error vectors")
    axis.grid(alpha=0.2)
    _save_figure(figure, path)


def build_parser() -> argparse.ArgumentParser:
    """Build the Milestone 9 command-line interface."""

    parser = argparse.ArgumentParser(
        prog="rf-trap-calibrated-validation",
        description="Run targeted local refinement and calibrated Data.txt diagnostics.",
    )
    parser.add_argument("--data", type=Path, default=Path("Data.txt"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation_results/milestone_9"),
    )
    parser.add_argument("--promotion-rows", type=int, default=20)
    parser.add_argument(
        "--maximum-estimated-central-triangles",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "--resume-interrupted-local",
        action="store_true",
        help="reproduce completed 500/200 um evidence and preserve prior failures",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run Milestone 9 and print the essential outcome and output paths."""

    arguments = build_parser().parse_args(argv)
    dataset = load_reference_dataset(arguments.data)
    study = run_milestone_9_study(
        dataset,
        promotion_row_count=arguments.promotion_rows,
        maximum_estimated_central_triangles=arguments.maximum_estimated_central_triangles,
        resume_interrupted_local=arguments.resume_interrupted_local,
        checkpoint_directory=arguments.output / "checkpoints",
    )
    paths = write_milestone_9_outputs(study, arguments.output)
    best = rank_calibration_evaluations(study.promoted_evaluations)[0].summary()
    print(f"best hypothesis: {best.hypothesis_name}")
    print(f"mean error: {1.0e3 * best.mean_error_m:.6g} mm")
    print(f"maximum error: {1.0e3 * best.maximum_error_m:.6g} mm")
    print(f"exactly-three rows: {best.exactly_three_rows}/{best.selected_rows}")
    print(f"validation gate passed: {best.validation_gate_passed}")
    print(f"report: {paths.report_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
