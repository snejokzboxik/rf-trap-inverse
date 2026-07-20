"""Milestone-7 analytic and implementation audits for the FEM pipeline."""

from __future__ import annotations

import argparse
import csv
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
from skfem import MeshTri

from .config import ForwardModelConfig, MeshConfig, SolverConfig
from .dataset import ReferenceDataset, load_reference_dataset
from .field import (
    element_electric_fields,
    recover_field,
    recover_nodal_electric_field,
)
from .forward import ForwardModelResult
from .geometry import GeometrySanity, TrapGeometry, build_geometry, geometry_sanity
from .mesh import (
    TrapMesh,
    generate_mesh,
    generate_perforated_disk_mesh,
    nearest_mesh_facet,
)
from .minima import MinimaSearchError, find_local_minima
from .real_scale import (
    REAL_ELECTRODE_RADIUS_M,
    REAL_INNER_RADIUS_M,
    REAL_OUTER_BOUNDARY_RADIUS_M,
    electrode_center_radius_m,
    real_scale_forward_config,
)
from .reference_validation import (
    ReferenceValidationReport,
    ReferenceValidationVariant,
    prepare_reference_row_inputs,
    run_reference_validation,
)
from .solver import FEMSolution, solve_dirichlet_laplace, solve_potential

ArtifactAction = Literal["flag", "filter"]


@dataclass(frozen=True)
class AnalyticErrorMetric:
    """One scalar error or invariant from an analytic FEM audit."""

    problem: str
    quantity: str
    value: float
    tolerance: float
    passed: bool
    sample_count: int
    units: str
    notes: str


@dataclass(frozen=True)
class ConcentricAuditResult:
    """Numerical and analytic arrays for the concentric-capacitor audit."""

    mesh: MeshTri
    numerical_potential_v: NDArray[np.float64]
    analytic_potential_v: NDArray[np.float64]
    numerical_field_v_per_m: NDArray[np.float64]
    analytic_field_v_per_m: NDArray[np.float64]
    field_mask: NDArray[np.bool_]
    metrics: tuple[AnalyticErrorMetric, ...]


@dataclass(frozen=True)
class UniformFieldAuditResult:
    """Exact-linear potential and recovered-field sign audit."""

    mesh: MeshTri
    numerical_potential_v: NDArray[np.float64]
    numerical_field_v_per_m: NDArray[np.float64]
    metrics: tuple[AnalyticErrorMetric, ...]


@dataclass(frozen=True)
class BoundaryMarkerDiagnostic:
    """Node-count, geometry, and imposed-value diagnostics for one boundary."""

    boundary: str
    node_count: int
    expected_potential_v: float
    maximum_geometry_residual_m: float
    maximum_potential_error_v: float
    overlap_node_count: int
    missing_boundary_node_count: int
    complete: bool


@dataclass(frozen=True)
class GeometrySanityDiagnostic:
    """Real-scale centres and clearances for one displacement case."""

    case: str
    row_number: int | None
    centers_m: NDArray[np.float64]
    electrode_radius_m: float
    outer_radius_m: float
    minimum_electrode_gap_m: float
    minimum_outer_clearance_m: float
    expected_center_error_m: float
    represented_as_circular_holes: bool
    valid: bool


@dataclass(frozen=True)
class SymmetryAuditResult:
    """D4 symmetry and central-minimum diagnostics for the undisplaced trap."""

    mesh_size_m: float
    center_field_magnitude_v_per_m: float
    potential_orbit_max_error_v: float
    field_equivariance_relative_error: float
    minimum_distance_from_center_m: float
    hessian_validated_candidates: int
    minimum_search_succeeded: bool
    metrics: tuple[AnalyticErrorMetric, ...]


@dataclass(frozen=True)
class MeshRefinementCase:
    """One reference-validation report at a stated mesh size and row set."""

    mesh_size_m: float
    row_numbers: tuple[int, ...]
    report: ReferenceValidationReport


@dataclass(frozen=True)
class MeshRefinementAssessment:
    """Pure summary of comparable full-row refinement cases."""

    coarse_mesh_size_m: float
    fine_mesh_size_m: float
    coarse_mean_error_m: float
    fine_mean_error_m: float
    relative_error_reduction: float
    coarse_exactly_three_rows: int
    fine_exactly_three_rows: int
    selected_rows: int
    meaningful_error_reduction: bool
    topology_improved_or_equal: bool
    topology_stable: bool


@dataclass(frozen=True)
class NumericalAuditStudy:
    """All Milestone-7 analytic, implementation, and refinement evidence."""

    concentric: ConcentricAuditResult
    uniform: UniformFieldAuditResult
    boundary_markers: tuple[BoundaryMarkerDiagnostic, ...]
    geometry_cases: tuple[GeometrySanityDiagnostic, ...]
    symmetry: SymmetryAuditResult
    refinement_cases: tuple[MeshRefinementCase, ...]
    refinement_assessment: MeshRefinementAssessment
    artifact_records: tuple[dict[str, object], ...]
    artifact_action: ArtifactAction
    runtime_seconds: float


@dataclass(frozen=True)
class NumericalAuditOutputPaths:
    """Paths written by the Milestone-7 numerical-audit serializer."""

    markdown_report: Path
    analytic_csv: Path
    boundary_csv: Path
    geometry_csv: Path
    refinement_csv: Path
    symmetry_csv: Path
    artifact_csv: Path
    plot_paths: tuple[Path, ...]


def concentric_potential_v(
    radius_m: ArrayLike,
    inner_radius_m: float,
    outer_radius_m: float,
) -> NDArray[np.float64]:
    """Return ``log(R/r) / log(R/a)`` for a concentric circular capacitor."""

    radius = np.asarray(radius_m, dtype=float)
    _validate_annulus_radii(radius, inner_radius_m, outer_radius_m)
    return np.log(outer_radius_m / radius) / np.log(
        outer_radius_m / inner_radius_m
    )


def concentric_electric_field_v_per_m(
    points_m: ArrayLike,
    inner_radius_m: float,
    outer_radius_m: float,
) -> NDArray[np.float64]:
    """Return the outward analytic field ``-grad(phi)`` in the annulus."""

    points = np.asarray(points_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("points_m must have finite shape (n, 2)")
    radius = np.linalg.norm(points, axis=1)
    _validate_annulus_radii(radius, inner_radius_m, outer_radius_m)
    denominator = radius * radius * np.log(outer_radius_m / inner_radius_m)
    return points / denominator[:, np.newaxis]


def run_concentric_capacitor_audit(
    *,
    inner_radius_m: float = 10.0e-3,
    outer_radius_m: float = 50.0e-3,
    mesh_size_m: float = 2.0e-3,
) -> ConcentricAuditResult:
    """Solve and compare the production P1 path with the annulus solution."""

    mesh_config = MeshConfig(characteristic_length_m=mesh_size_m)
    annulus = generate_perforated_disk_mesh(
        outer_radius_m=outer_radius_m,
        hole_centers_m=((0.0, 0.0),),
        hole_radii_m=(inner_radius_m,),
        config=mesh_config,
    )
    prescribed = np.zeros(annulus.mesh.p.shape[1], dtype=float)
    prescribed[annulus.hole_boundary_nodes[0]] = 1.0
    prescribed[annulus.outer_boundary_nodes] = 0.0
    boundary_nodes = np.union1d(
        annulus.hole_boundary_nodes[0],
        annulus.outer_boundary_nodes,
    )
    solution = solve_dirichlet_laplace(
        annulus.mesh,
        boundary_nodes,
        prescribed,
        SolverConfig(),
    )
    points = annulus.mesh.p.T
    radius = np.linalg.norm(points, axis=1)
    analytic_potential = concentric_potential_v(
        radius,
        inner_radius_m,
        outer_radius_m,
    )
    free = np.setdiff1d(np.arange(points.shape[0]), boundary_nodes)
    potential_error = solution.potential_v[free] - analytic_potential[free]
    potential_relative = _relative_l2(
        solution.potential_v[free],
        analytic_potential[free],
    )

    numerical_field = recover_nodal_electric_field(
        annulus.mesh,
        solution.potential_v,
    ).T
    analytic_field = concentric_electric_field_v_per_m(
        points,
        inner_radius_m,
        outer_radius_m,
    )
    field_mask = (radius >= inner_radius_m + 1.5 * mesh_size_m) & (
        radius <= outer_radius_m - 1.5 * mesh_size_m
    )
    field_relative = _relative_l2(
        numerical_field[field_mask],
        analytic_field[field_mask],
    )
    magnitude_relative = _relative_l2(
        np.linalg.norm(numerical_field[field_mask], axis=1),
        np.linalg.norm(analytic_field[field_mask], axis=1),
    )

    raw_field = element_electric_fields(annulus.mesh, solution.potential_v).T
    centroids = np.mean(annulus.mesh.p[:, annulus.mesh.t], axis=1).T
    centroid_radius = np.linalg.norm(centroids, axis=1)
    raw_mask = (centroid_radius >= inner_radius_m + mesh_size_m) & (
        centroid_radius <= outer_radius_m - mesh_size_m
    )
    analytic_raw = concentric_electric_field_v_per_m(
        centroids,
        inner_radius_m,
        outer_radius_m,
    )
    raw_relative = _relative_l2(raw_field[raw_mask], analytic_raw[raw_mask])
    radial_components = np.einsum(
        "ij,ij->i",
        numerical_field[field_mask],
        points[field_mask] / radius[field_mask, np.newaxis],
    )
    metrics = (
        _metric(
            "concentric-capacitor",
            "potential-relative-l2",
            potential_relative,
            0.02,
            len(free),
            "1",
            "free P1 vertices",
        ),
        _metric(
            "concentric-capacitor",
            "potential-maximum-absolute-error",
            float(np.max(np.abs(potential_error))),
            0.03,
            len(free),
            "V",
            "free P1 vertices",
        ),
        _metric(
            "concentric-capacitor",
            "recovered-field-relative-l2",
            field_relative,
            0.10,
            int(np.count_nonzero(field_mask)),
            "1",
            "nodes at least 1.5 mesh lengths from both boundaries",
        ),
        _metric(
            "concentric-capacitor",
            "recovered-field-magnitude-relative-l2",
            magnitude_relative,
            0.10,
            int(np.count_nonzero(field_mask)),
            "1",
            "magnitude comparison on the recovered-field mask",
        ),
        _metric(
            "concentric-capacitor",
            "raw-element-field-relative-l2",
            raw_relative,
            0.10,
            int(np.count_nonzero(raw_mask)),
            "1",
            "element centroids at least one mesh length from boundaries",
        ),
        _metric(
            "concentric-capacitor",
            "minimum-outward-radial-field",
            float(np.min(radial_components)),
            0.0,
            len(radial_components),
            "V/m",
            "pass means every audited recovered field points outward",
            greater_than=True,
        ),
        _metric(
            "concentric-capacitor",
            "relative-free-residual",
            solution.relative_free_residual,
            1.0e-10,
            len(free),
            "1",
            "production generic Dirichlet solver",
        ),
        _metric(
            "concentric-capacitor",
            "Dirichlet-boundary-error",
            solution.boundary_error_v,
            1.0e-12,
            len(boundary_nodes),
            "V",
            "inner 1 V and outer 0 V nodes",
        ),
    )
    return ConcentricAuditResult(
        mesh=annulus.mesh,
        numerical_potential_v=solution.potential_v,
        analytic_potential_v=analytic_potential,
        numerical_field_v_per_m=numerical_field,
        analytic_field_v_per_m=analytic_field,
        field_mask=field_mask,
        metrics=metrics,
    )


def run_uniform_field_audit(*, refinements: int = 4) -> UniformFieldAuditResult:
    """Verify an exact linear potential and the sign of ``E = -grad(phi)``."""

    if refinements < 1:
        raise ValueError("refinements must be positive")
    mesh = MeshTri.init_circle(nrefs=refinements)
    boundary = mesh.boundary_nodes()
    prescribed = mesh.p[0].copy()
    solution = solve_dirichlet_laplace(
        mesh,
        boundary,
        prescribed,
        SolverConfig(),
    )
    numerical_field = recover_nodal_electric_field(mesh, solution.potential_v).T
    expected_field = np.tile((-1.0, 0.0), (mesh.p.shape[1], 1))
    potential_max = float(np.max(np.abs(solution.potential_v - mesh.p[0])))
    field_max = float(np.max(np.linalg.norm(numerical_field - expected_field, axis=1)))
    mean_x = float(np.mean(numerical_field[:, 0]))
    metrics = (
        _metric(
            "uniform-field-disk",
            "potential-maximum-absolute-error",
            potential_max,
            1.0e-11,
            mesh.p.shape[1],
            "V",
            "analytic phi=x",
        ),
        _metric(
            "uniform-field-disk",
            "field-maximum-vector-error",
            field_max,
            1.0e-10,
            mesh.p.shape[1],
            "V/m",
            "expected E=(-1,0)",
        ),
        _metric(
            "uniform-field-disk",
            "mean-electric-field-x",
            mean_x,
            -0.999999999,
            mesh.p.shape[1],
            "V/m",
            "pass requires negative Ex and fails for +grad(phi)",
            less_than=True,
        ),
    )
    return UniformFieldAuditResult(
        mesh=mesh,
        numerical_potential_v=solution.potential_v,
        numerical_field_v_per_m=numerical_field,
        metrics=metrics,
    )


def audit_boundary_markers(
    geometry: TrapGeometry,
    trap_mesh: TrapMesh,
    solution: FEMSolution,
) -> tuple[BoundaryMarkerDiagnostic, ...]:
    """Report completeness, exclusivity, geometry, and values of every marker."""

    groups = (
        ("outer", trap_mesh.outer_boundary_nodes, np.zeros(2), geometry.config.outer_radius_m, geometry.config.outer_potential_v),
        *tuple(
            (
                f"electrode-{index}",
                nodes,
                center,
                geometry.config.electrode_radius_m,
                potential,
            )
            for index, (nodes, center, potential) in enumerate(
                zip(
                    trap_mesh.electrode_boundary_nodes_by_electrode,
                    geometry.centers_m,
                    geometry.config.resolved_electrode_potentials_v,
                    strict=True,
                ),
                start=1,
            )
        ),
    )
    all_boundary = trap_mesh.mesh.boundary_nodes()
    concatenated = np.concatenate([group[1] for group in groups])
    unique = np.unique(concatenated)
    overlap_count = int(concatenated.size - unique.size)
    missing_count = int(np.setdiff1d(all_boundary, unique).size)
    records = []
    for label, nodes, center, radius, expected in groups:
        residual = np.abs(
            np.linalg.norm(trap_mesh.mesh.p[:, nodes].T - center, axis=1) - radius
        )
        records.append(
            BoundaryMarkerDiagnostic(
                boundary=label,
                node_count=int(nodes.size),
                expected_potential_v=float(expected),
                maximum_geometry_residual_m=float(np.max(residual)),
                maximum_potential_error_v=float(
                    np.max(np.abs(solution.potential_v[nodes] - expected))
                ),
                overlap_node_count=overlap_count,
                missing_boundary_node_count=missing_count,
                complete=(
                    nodes.size > 0
                    and overlap_count == 0
                    and missing_count == 0
                ),
            )
        )
    return tuple(records)


def build_geometry_sanity_diagnostics(
    dataset: ReferenceDataset,
    config: ForwardModelConfig,
    row_numbers: Sequence[int] = tuple(range(1, 11)),
) -> tuple[GeometrySanityDiagnostic, ...]:
    """Audit nominal and dataset-displaced real-scale circular-hole geometry."""

    variant = _audit_variant()
    nominal = build_geometry(config.geometry, np.zeros(6))
    records = [_geometry_record("nominal", None, nominal, nominal.centers_m)]
    for row_number in row_numbers:
        row_index = row_number - 1
        solver_displacements, _, row_config = prepare_reference_row_inputs(
            dataset.raw_displacements_m[row_index],
            dataset.raw_minima_absolute_m[row_index],
            config,
            variant,
        )
        geometry = build_geometry(row_config.geometry, solver_displacements)
        expected = np.asarray(row_config.geometry.nominal_centers_m, dtype=float).copy()
        expected[1:] += solver_displacements.reshape(3, 2)
        records.append(
            _geometry_record(
                f"reference-row-{row_number}",
                row_number,
                geometry,
                expected,
            )
        )
    return tuple(records)


def run_symmetric_trap_audit(
    geometry: TrapGeometry,
    trap_mesh: TrapMesh,
    solution: FEMSolution,
    config: ForwardModelConfig,
) -> SymmetryAuditResult:
    """Check central field, rotational equivariance, and central minimum."""

    field = recover_field(solution)
    center_field = np.asarray(field.evaluate((0.0, 0.0)), dtype=float)
    potential_interpolator = solution.basis.interpolator(solution.potential_v)
    rotation = np.asarray(((0.0, -1.0), (1.0, 0.0)))
    potential_error = 0.0
    field_error = 0.0
    field_scale = 0.0
    for radius in (2.0e-3, 4.0e-3, 6.0e-3):
        base = np.asarray([radius, 0.0])
        points = []
        current = base
        for _ in range(4):
            points.append(current)
            current = rotation @ current
        orbit = np.asarray(points)
        potentials = np.asarray(potential_interpolator(orbit.T), dtype=float)
        fields = np.asarray(field.evaluate(orbit), dtype=float)
        potential_error = max(potential_error, float(np.ptp(potentials)))
        for index in range(1, 4):
            expected = np.linalg.matrix_power(rotation, index) @ fields[0]
            field_error = max(field_error, float(np.linalg.norm(fields[index] - expected)))
        field_scale = max(field_scale, float(np.max(np.linalg.norm(fields, axis=1))))
    field_relative = field_error / max(field_scale, np.finfo(float).tiny)

    minimum_succeeded = True
    minimum_distance = float("nan")
    validated_count = 0
    try:
        minima, diagnostics = find_local_minima(
            field,
            replace(config.minima, expected_minima=1),
        )
        minimum_distance = float(np.linalg.norm(minima[0].position_m))
        validated_count = diagnostics.hessian_validated_candidates
    except MinimaSearchError:
        minimum_succeeded = False
    center_magnitude = float(np.linalg.norm(center_field))
    metrics = (
        _metric(
            "symmetric-four-electrode",
            "center-field-relative-to-probe-field",
            center_magnitude / max(field_scale, np.finfo(float).tiny),
            0.05,
            1,
            "1",
            "zero follows from D4 symmetry; tolerance includes recovered-gradient error",
        ),
        _metric(
            "symmetric-four-electrode",
            "potential-orbit-maximum-error",
            potential_error,
            0.01,
            12,
            "V",
            "fourfold rotations at radii 2, 4, and 6 mm",
        ),
        _metric(
            "symmetric-four-electrode",
            "field-rotation-relative-error",
            field_relative,
            0.10,
            12,
            "1",
            "recovered-field equivariance under 90-degree rotation",
        ),
        _metric(
            "symmetric-four-electrode",
            "central-minimum-distance",
            minimum_distance,
            config.mesh.characteristic_length_m,
            1,
            "m",
            "one-minimum diagnostic; default expected-three physics is unchanged",
        ),
    )
    return SymmetryAuditResult(
        mesh_size_m=config.mesh.characteristic_length_m,
        center_field_magnitude_v_per_m=center_magnitude,
        potential_orbit_max_error_v=potential_error,
        field_equivariance_relative_error=field_relative,
        minimum_distance_from_center_m=minimum_distance,
        hessian_validated_candidates=validated_count,
        minimum_search_succeeded=minimum_succeeded,
        metrics=metrics,
    )


def diagnose_recovered_minima(
    result: ForwardModelResult,
    *,
    artifact_action: ArtifactAction = "flag",
    facet_fraction_threshold: float = 0.02,
    jump_ratio_threshold: float = 0.50,
    psi_ratio_threshold: float = 100.0,
) -> tuple[dict[str, object], ...]:
    """Flag or audit-filter candidates using documented recovered-field criteria.

    A candidate is flagged when it is both facet-locked and adjacent raw P1
    fields have a large relative jump, or when its recovered ``|E|^2`` exceeds
    the lowest validated candidate by ``psi_ratio_threshold``. ``filter`` only
    changes the reported ``retained_after_artifact_screen`` field; it never
    changes the forward API or silently discards a physical result.
    """

    if artifact_action not in ("flag", "filter"):
        raise ValueError("artifact_action must be 'flag' or 'filter'")
    thresholds = np.asarray(
        (facet_fraction_threshold, jump_ratio_threshold, psi_ratio_threshold),
        dtype=float,
    )
    if not np.all(np.isfinite(thresholds)) or np.any(thresholds <= 0.0):
        raise ValueError("artifact thresholds must be finite and positive")
    candidates = result.minima_diagnostics.hessian_validated_minima
    if not candidates:
        return ()
    mesh = result.trap_mesh.mesh
    raw_fields = element_electric_fields(mesh, result.fem_solution.potential_v)
    minimum_psi = min(item.pseudopotential_v2_per_m2 for item in candidates)
    psi_floor = max(minimum_psi, np.finfo(float).tiny)
    finder = mesh.element_finder()
    records = []
    for rank, candidate in enumerate(candidates, start=1):
        distance, facet_index = nearest_mesh_facet(mesh, candidate.position_m)
        adjacent = mesh.f2t[:, facet_index]
        adjacent = adjacent[adjacent >= 0]
        adjacent_fields = raw_fields[:, adjacent]
        if adjacent.size == 2:
            jump = float(np.linalg.norm(adjacent_fields[:, 0] - adjacent_fields[:, 1]))
            local_scale = float(
                max(
                    np.linalg.norm(adjacent_fields[:, 0]),
                    np.linalg.norm(adjacent_fields[:, 1]),
                    np.finfo(float).tiny,
                )
            )
            jump_ratio = jump / local_scale
        else:
            jump = float("nan")
            jump_ratio = float("nan")
        triangle = int(
            finder(
                np.asarray([candidate.position_m[0]]),
                np.asarray([candidate.position_m[1]]),
            )[0]
        )
        raw_field = raw_fields[:, triangle]
        raw_magnitude = float(np.linalg.norm(raw_field))
        recovered_magnitude = float(np.sqrt(candidate.pseudopotential_v2_per_m2))
        facet_fraction = distance / result.trap_mesh.mesh.param()
        psi_ratio = candidate.pseudopotential_v2_per_m2 / psi_floor
        facet_locked = bool(
            facet_fraction <= facet_fraction_threshold
            and np.isfinite(jump_ratio)
            and jump_ratio >= jump_ratio_threshold
        )
        high_psi = bool(psi_ratio >= psi_ratio_threshold)
        flagged = facet_locked or high_psi
        selected = any(
            np.array_equal(candidate.position_m, minimum.position_m)
            for minimum in result.minima
        )
        reasons = []
        if facet_locked:
            reasons.append("facet-locked-large-raw-field-jump")
        if high_psi:
            reasons.append("high-recovered-psi-relative-to-best-candidate")
        records.append(
            {
                "candidate_rank_by_psi": rank,
                "x_m": candidate.position_m[0],
                "y_m": candidate.position_m[1],
                "recovered_psi_v2_per_m2": candidate.pseudopotential_v2_per_m2,
                "recovered_field_magnitude_v_per_m": recovered_magnitude,
                "raw_element_field_magnitude_v_per_m": raw_magnitude,
                "raw_element_psi_v2_per_m2": raw_magnitude * raw_magnitude,
                "raw_to_recovered_field_ratio": raw_magnitude
                / max(recovered_magnitude, np.finfo(float).tiny),
                "nearest_facet_distance_m": distance,
                "nearest_facet_distance_over_mesh_h": facet_fraction,
                "adjacent_raw_field_jump_v_per_m": jump,
                "adjacent_raw_field_jump_ratio": jump_ratio,
                "recovered_psi_ratio_to_best_candidate": psi_ratio,
                "selected_by_forward_api": selected,
                "artifact_flagged": flagged,
                "artifact_reasons": ";".join(reasons),
                "artifact_action": artifact_action,
                "retained_after_artifact_screen": (
                    True if artifact_action == "flag" else not flagged
                ),
            }
        )
    return tuple(records)


def assess_mesh_refinement(
    cases: Sequence[MeshRefinementCase],
    *,
    meaningful_reduction_fraction: float = 0.05,
) -> MeshRefinementAssessment:
    """Compare coarsest and finest cases sharing the largest identical row set."""

    if not cases or not 0.0 < meaningful_reduction_fraction < 1.0:
        raise ValueError("cases and a fractional reduction threshold are required")
    largest = max(len(case.row_numbers) for case in cases)
    comparable = [case for case in cases if len(case.row_numbers) == largest]
    row_set = comparable[0].row_numbers
    if any(case.row_numbers != row_set for case in comparable):
        raise ValueError("full-row refinement cases must use the same ordered rows")
    coarse = max(comparable, key=lambda case: case.mesh_size_m)
    fine = min(comparable, key=lambda case: case.mesh_size_m)
    coarse_summary = coarse.report.summary()
    fine_summary = fine.report.summary()
    reduction = 1.0 - fine_summary.mean_error_m / coarse_summary.mean_error_m
    topology_nondecreasing = (
        fine_summary.rows_with_exactly_three_physical_minima
        >= coarse_summary.rows_with_exactly_three_physical_minima
    )
    return MeshRefinementAssessment(
        coarse_mesh_size_m=coarse.mesh_size_m,
        fine_mesh_size_m=fine.mesh_size_m,
        coarse_mean_error_m=coarse_summary.mean_error_m,
        fine_mean_error_m=fine_summary.mean_error_m,
        relative_error_reduction=reduction,
        coarse_exactly_three_rows=(
            coarse_summary.rows_with_exactly_three_physical_minima
        ),
        fine_exactly_three_rows=fine_summary.rows_with_exactly_three_physical_minima,
        selected_rows=largest,
        meaningful_error_reduction=(
            np.isfinite(reduction) and reduction >= meaningful_reduction_fraction
        ),
        topology_improved_or_equal=topology_nondecreasing,
        topology_stable=(
            topology_nondecreasing
            and fine_summary.rows_with_exactly_three_physical_minima == largest
        ),
    )


def run_reference_mesh_refinement(
    dataset: ReferenceDataset,
    *,
    mesh_sizes_m: Sequence[float] = (2.0e-3, 1.5e-3, 1.0e-3, 0.75e-3),
    row_numbers: Sequence[int] = tuple(range(1, 11)),
    include_half_mm_rows: Sequence[int] = (),
) -> tuple[MeshRefinementCase, ...]:
    """Run coherent E1-relative E2/E3-swapped validation in fresh processes."""

    meshes = tuple(float(value) for value in mesh_sizes_m)
    rows = tuple(int(value) for value in row_numbers)
    if not meshes or any(not np.isfinite(value) or value <= 0.0 for value in meshes):
        raise ValueError("mesh_sizes_m must contain finite positive values")
    if not rows:
        raise ValueError("row_numbers must be nonempty")
    variant = _audit_variant()
    cases = []
    for mesh_size in meshes:
        report = run_reference_validation(
            dataset,
            real_scale_forward_config(mesh_size_m=mesh_size),
            rows,
            variant=variant,
        )
        cases.append(MeshRefinementCase(mesh_size, rows, report))
    half_rows = tuple(int(value) for value in include_half_mm_rows)
    if half_rows:
        report = run_reference_validation(
            dataset,
            real_scale_forward_config(mesh_size_m=0.5e-3),
            half_rows,
            variant=variant,
        )
        cases.append(MeshRefinementCase(0.5e-3, half_rows, report))
    return tuple(cases)


def run_fem_numerical_audit(
    dataset: ReferenceDataset,
    *,
    analytic_mesh_size_m: float = 2.0e-3,
    refinement_mesh_sizes_m: Sequence[float] = (2.0e-3, 1.5e-3, 1.0e-3, 0.75e-3),
    include_half_mm: bool = True,
    artifact_action: ArtifactAction = "flag",
) -> NumericalAuditStudy:
    """Run the complete Milestone-7 audit without changing default physics."""

    started = time.perf_counter()
    concentric = run_concentric_capacitor_audit(mesh_size_m=analytic_mesh_size_m)
    uniform = run_uniform_field_audit()
    real_config = real_scale_forward_config(mesh_size_m=analytic_mesh_size_m)
    geometry = build_geometry(real_config.geometry, np.zeros(6))
    trap_mesh = generate_mesh(geometry, real_config.mesh)
    solution = solve_potential(geometry, trap_mesh, real_config.solver)
    boundary = audit_boundary_markers(geometry, trap_mesh, solution)
    geometry_cases = build_geometry_sanity_diagnostics(dataset, real_config)
    symmetry_config = real_scale_forward_config(
        mesh_size_m=min(analytic_mesh_size_m, 1.0e-3)
    )
    symmetry_geometry = build_geometry(symmetry_config.geometry, np.zeros(6))
    symmetry_mesh = generate_mesh(symmetry_geometry, symmetry_config.mesh)
    symmetry_solution = solve_potential(
        symmetry_geometry,
        symmetry_mesh,
        symmetry_config.solver,
    )
    symmetry = run_symmetric_trap_audit(
        symmetry_geometry,
        symmetry_mesh,
        symmetry_solution,
        symmetry_config,
    )
    artifact_records = _run_artifact_rows(
        dataset,
        real_config,
        tuple(range(1, 11)),
        artifact_action,
    )
    refinement_cases = run_reference_mesh_refinement(
        dataset,
        mesh_sizes_m=refinement_mesh_sizes_m,
        include_half_mm_rows=(1, 2, 3) if include_half_mm else (),
    )
    assessment = assess_mesh_refinement(refinement_cases)
    return NumericalAuditStudy(
        concentric=concentric,
        uniform=uniform,
        boundary_markers=boundary,
        geometry_cases=geometry_cases,
        symmetry=symmetry,
        refinement_cases=refinement_cases,
        refinement_assessment=assessment,
        artifact_records=artifact_records,
        artifact_action=artifact_action,
        runtime_seconds=time.perf_counter() - started,
    )


def write_fem_numerical_audit_outputs(
    study: NumericalAuditStudy,
    output_directory: str | Path,
) -> NumericalAuditOutputPaths:
    """Write requested CSV, Markdown, and plot artifacts for one audit study."""

    output = Path(output_directory)
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    paths = NumericalAuditOutputPaths(
        markdown_report=output / "fem_analytic_audit_report.md",
        analytic_csv=output / "analytic_error_summary.csv",
        boundary_csv=output / "boundary_marker_summary.csv",
        geometry_csv=output / "geometry_sanity_summary.csv",
        refinement_csv=output / "mesh_refinement_reference_validation.csv",
        symmetry_csv=output / "symmetry_audit_summary.csv",
        artifact_csv=output / "minima_interpolation_diagnostics.csv",
        plot_paths=(
            plots / "concentric_potential.png",
            plots / "concentric_field_magnitude.png",
            plots / "uniform_field_error.png",
            plots / "mesh_refinement_error.png",
            plots / "mesh_refinement_topology.png",
            plots / "minima_artifact_diagnostics.png",
        ),
    )
    analytic_metrics = (
        study.concentric.metrics + study.uniform.metrics + study.symmetry.metrics
    )
    _write_csv(paths.analytic_csv, [_analytic_record(item) for item in analytic_metrics])
    _write_csv(paths.boundary_csv, [_boundary_record(item) for item in study.boundary_markers])
    _write_csv(paths.geometry_csv, [_geometry_csv_record(item) for item in study.geometry_cases])
    _write_csv(paths.refinement_csv, _refinement_records(study.refinement_cases))
    _write_csv(paths.symmetry_csv, [_symmetry_record(study.symmetry)])
    _write_csv(paths.artifact_csv, list(study.artifact_records))
    paths.markdown_report.write_text(_markdown_report(study), encoding="utf-8")
    _plot_concentric_potential(study.concentric, paths.plot_paths[0])
    _plot_concentric_field(study.concentric, paths.plot_paths[1])
    _plot_uniform_field(study.uniform, paths.plot_paths[2])
    _plot_refinement_error(study.refinement_cases, paths.plot_paths[3])
    _plot_refinement_topology(study.refinement_cases, paths.plot_paths[4])
    _plot_artifacts(study.artifact_records, paths.plot_paths[5])
    return paths


def _validate_annulus_radii(
    radius_m: NDArray[np.float64],
    inner_radius_m: float,
    outer_radius_m: float,
) -> None:
    if (
        not np.isfinite(inner_radius_m)
        or not np.isfinite(outer_radius_m)
        or inner_radius_m <= 0.0
        or outer_radius_m <= inner_radius_m
        or not np.all(np.isfinite(radius_m))
        or np.any(radius_m < inner_radius_m - 1.0e-12)
        or np.any(radius_m > outer_radius_m + 1.0e-12)
    ):
        raise ValueError("radii must lie inside a finite positive annulus")


def _relative_l2(numerical: ArrayLike, analytic: ArrayLike) -> float:
    numerical_array = np.asarray(numerical, dtype=float)
    analytic_array = np.asarray(analytic, dtype=float)
    return float(
        np.linalg.norm(numerical_array - analytic_array)
        / max(np.linalg.norm(analytic_array), np.finfo(float).tiny)
    )


def _metric(
    problem: str,
    quantity: str,
    value: float,
    tolerance: float,
    sample_count: int,
    units: str,
    notes: str,
    *,
    greater_than: bool = False,
    less_than: bool = False,
) -> AnalyticErrorMetric:
    if greater_than and less_than:
        raise ValueError("a metric cannot use both one-sided comparisons")
    if greater_than:
        passed = bool(np.isfinite(value) and value > tolerance)
    elif less_than:
        passed = bool(np.isfinite(value) and value < tolerance)
    else:
        passed = bool(np.isfinite(value) and value <= tolerance)
    return AnalyticErrorMetric(
        problem=problem,
        quantity=quantity,
        value=value,
        tolerance=tolerance,
        passed=passed,
        sample_count=sample_count,
        units=units,
        notes=notes,
    )


def _geometry_record(
    case: str,
    row_number: int | None,
    geometry: TrapGeometry,
    expected_centers_m: NDArray[np.float64],
) -> GeometrySanityDiagnostic:
    sanity: GeometrySanity = geometry_sanity(geometry.config, geometry.centers_m)
    return GeometrySanityDiagnostic(
        case=case,
        row_number=row_number,
        centers_m=geometry.centers_m.copy(),
        electrode_radius_m=geometry.config.electrode_radius_m,
        outer_radius_m=geometry.config.outer_radius_m,
        minimum_electrode_gap_m=sanity.minimum_electrode_gap_m,
        minimum_outer_clearance_m=sanity.minimum_outer_clearance_m,
        expected_center_error_m=float(
            np.max(np.abs(geometry.centers_m - expected_centers_m))
        ),
        represented_as_circular_holes=True,
        valid=sanity.valid,
    )


def _audit_variant() -> ReferenceValidationVariant:
    return ReferenceValidationVariant(
        name="m7_relative_all_positive_perm1324",
        displacement_mode="electrode1-relative",
        electrode_permutation=(1, 3, 2, 4),
        polarity_name="all-positive",
    )


def _run_artifact_rows(
    dataset: ReferenceDataset,
    config: ForwardModelConfig,
    row_numbers: Sequence[int],
    artifact_action: ArtifactAction,
) -> tuple[dict[str, object], ...]:
    records = []
    variant = _audit_variant()
    for row_number in row_numbers:
        index = row_number - 1
        displacements, _, row_config = prepare_reference_row_inputs(
            dataset.raw_displacements_m[index],
            dataset.raw_minima_absolute_m[index],
            config,
            variant,
        )
        outcome = _run_artifact_worker(displacements, row_config, artifact_action)
        if outcome.get("ok"):
            for record in outcome["records"]:
                records.append(
                    {
                        "row_number": row_number,
                        "status": "ok",
                        "error_type": "",
                        "error_message": "",
                        **record,
                    }
                )
        else:
            records.append(
                {
                    "row_number": row_number,
                    "status": "failed",
                    "error_type": outcome.get("error_type", "unknown"),
                    "error_message": outcome.get("error_message", "unknown"),
                    "candidate_rank_by_psi": "",
                    "artifact_flagged": "",
                    "artifact_action": artifact_action,
                    "retained_after_artifact_screen": "",
                }
            )
    return tuple(records)


def _run_artifact_worker(
    displacements_m: NDArray[np.float64],
    config: ForwardModelConfig,
    artifact_action: ArtifactAction,
) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-m", "rf_trap_forward._fem_audit_worker"],
        input=pickle.dumps((displacements_m, config, artifact_action)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "ok": False,
            "error_type": "AuditWorkerProcessError",
            "error_message": completed.stderr.decode("utf-8", errors="replace").strip(),
        }
    try:
        outcome = pickle.loads(completed.stdout)
    except Exception as error:
        return {
            "ok": False,
            "error_type": "AuditWorkerProtocolError",
            "error_message": str(error),
        }
    if not isinstance(outcome, dict) or "ok" not in outcome:
        return {
            "ok": False,
            "error_type": "AuditWorkerProtocolError",
            "error_message": "worker returned an invalid object",
        }
    return outcome


def _analytic_record(metric: AnalyticErrorMetric) -> dict[str, object]:
    return {
        "problem": metric.problem,
        "quantity": metric.quantity,
        "value": metric.value,
        "tolerance": metric.tolerance,
        "passed": metric.passed,
        "sample_count": metric.sample_count,
        "units": metric.units,
        "notes": metric.notes,
    }


def _boundary_record(item: BoundaryMarkerDiagnostic) -> dict[str, object]:
    return {
        "boundary": item.boundary,
        "node_count": item.node_count,
        "expected_potential_v": item.expected_potential_v,
        "maximum_geometry_residual_m": item.maximum_geometry_residual_m,
        "maximum_potential_error_v": item.maximum_potential_error_v,
        "overlap_node_count": item.overlap_node_count,
        "missing_boundary_node_count": item.missing_boundary_node_count,
        "complete": item.complete,
    }


def _geometry_csv_record(item: GeometrySanityDiagnostic) -> dict[str, object]:
    record: dict[str, object] = {
        "case": item.case,
        "row_number": "" if item.row_number is None else item.row_number,
        "electrode_radius_m": item.electrode_radius_m,
        "outer_radius_m": item.outer_radius_m,
        "minimum_electrode_gap_m": item.minimum_electrode_gap_m,
        "minimum_outer_clearance_m": item.minimum_outer_clearance_m,
        "expected_center_error_m": item.expected_center_error_m,
        "represented_as_circular_holes": item.represented_as_circular_holes,
        "valid": item.valid,
    }
    for index, center in enumerate(item.centers_m, start=1):
        record[f"electrode_{index}_center_x_m"] = center[0]
        record[f"electrode_{index}_center_y_m"] = center[1]
    return record


def _refinement_records(cases: Sequence[MeshRefinementCase]) -> list[dict[str, object]]:
    records = []
    for case in cases:
        summary = case.report.summary()
        records.append(
            {
                "record_type": "summary",
                "mesh_size_m": case.mesh_size_m,
                "row_number": "",
                "selected_rows": summary.selected_rows,
                "completed_rows": summary.completed_rows,
                "failed_rows": summary.failed_rows,
                "rows_exactly_three_physical_minima": summary.rows_with_exactly_three_physical_minima,
                "matched_minima": summary.matched_minima,
                "mean_error_m": summary.mean_error_m,
                "median_error_m": summary.median_error_m,
                "maximum_error_m": summary.maximum_error_m,
                "percentile_95_error_m": summary.percentile_95_error_m,
                "row_status": "",
                "row_hessian_validated_candidates": "",
                "row_mean_error_m": "",
                "row_maximum_error_m": "",
                "runtime_seconds": case.report.runtime_seconds,
                "error_type": "",
                "error_message": "",
            }
        )
        for row in case.report.rows:
            errors = row.error_distances_m()
            records.append(
                {
                    "record_type": "row",
                    "mesh_size_m": case.mesh_size_m,
                    "row_number": row.row_number,
                    "selected_rows": "",
                    "completed_rows": "",
                    "failed_rows": "",
                    "rows_exactly_three_physical_minima": "",
                    "matched_minima": len(row.matches),
                    "mean_error_m": "",
                    "median_error_m": "",
                    "maximum_error_m": "",
                    "percentile_95_error_m": "",
                    "row_status": row.status,
                    "row_hessian_validated_candidates": (
                        ""
                        if row.observation is None
                        else row.observation.hessian_validated_candidates
                    ),
                    "row_mean_error_m": (
                        "" if errors.size == 0 else float(np.mean(errors))
                    ),
                    "row_maximum_error_m": (
                        "" if errors.size == 0 else float(np.max(errors))
                    ),
                    "runtime_seconds": row.runtime_seconds,
                    "error_type": row.error_type,
                    "error_message": row.error_message,
                }
            )
    return records


def _mean_error_for_rows(
    report: ReferenceValidationReport,
    row_numbers: Sequence[int],
) -> float:
    selected = set(row_numbers)
    errors = np.asarray(
        [
            match.distance_m
            for row in report.rows
            if row.row_number in selected
            for match in row.matches
        ],
        dtype=float,
    )
    if errors.size == 0:
        return float("nan")
    return float(np.mean(errors))


def _symmetry_record(item: SymmetryAuditResult) -> dict[str, object]:
    return {
        "mesh_size_m": item.mesh_size_m,
        "center_field_magnitude_v_per_m": item.center_field_magnitude_v_per_m,
        "potential_orbit_max_error_v": item.potential_orbit_max_error_v,
        "field_equivariance_relative_error": item.field_equivariance_relative_error,
        "minimum_distance_from_center_m": item.minimum_distance_from_center_m,
        "hessian_validated_candidates": item.hessian_validated_candidates,
        "minimum_search_succeeded": item.minimum_search_succeeded,
        "all_symmetry_metrics_passed": all(metric.passed for metric in item.metrics),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty audit table: {path.name}")
    fieldnames = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_report(study: NumericalAuditStudy) -> str:
    analytic = study.concentric.metrics + study.uniform.metrics
    analytic_pass = all(metric.passed for metric in analytic)
    boundary_pass = all(item.complete for item in study.boundary_markers)
    geometry_pass = all(item.valid for item in study.geometry_cases)
    symmetry_pass = all(metric.passed for metric in study.symmetry.metrics)
    core_numerics_pass = (
        analytic_pass and boundary_pass and geometry_pass and symmetry_pass
    )
    assessment = study.refinement_assessment
    flagged = sum(record.get("artifact_flagged") is True for record in study.artifact_records)
    candidate_records = sum(record.get("status") == "ok" for record in study.artifact_records)
    selected_records = sum(
        record.get("selected_by_forward_api") is True
        for record in study.artifact_records
    )
    flagged_selected = sum(
        record.get("selected_by_forward_api") is True
        and record.get("artifact_flagged") is True
        for record in study.artifact_records
    )
    flagged_selected_fraction = flagged_selected / max(selected_records, 1)
    artifacts_dominate_selected = flagged_selected_fraction > 0.10
    model_class_conclusion_justified = (
        core_numerics_pass
        and not assessment.meaningful_error_reduction
        and not artifacts_dominate_selected
    )
    partial_cases = [
        case
        for case in study.refinement_cases
        if len(case.row_numbers) < assessment.selected_rows
    ]
    partial_refinement_lines = []
    if partial_cases:
        partial = min(partial_cases, key=lambda case: case.mesh_size_m)
        coarse_full = max(
            (
                case
                for case in study.refinement_cases
                if len(case.row_numbers) == assessment.selected_rows
            ),
            key=lambda case: case.mesh_size_m,
        )
        coarse_partial_mean = _mean_error_for_rows(
            coarse_full.report,
            partial.row_numbers,
        )
        fine_partial_mean = partial.report.summary().mean_error_m
        partial_reduction = 1.0 - fine_partial_mean / coarse_partial_mean
        partial_refinement_lines = [
            f"For the optional rows {partial.row_numbers[0]}--{partial.row_numbers[-1]}",
            f"h={partial.mesh_size_m * 1e3:.6g} mm check, mean error changes from",
            f"{_mm(coarse_partial_mean)} mm at h={coarse_full.mesh_size_m * 1e3:.6g} mm",
            f"to {_mm(fine_partial_mean)} mm, a `{partial_reduction * 100.0:.3f}%` reduction.",
        ]
    lines = [
        "# Milestone 7: FEM numerical audit",
        "",
        "## Scope",
        "",
        "The audit preserves the all-positive four-electrode default physics.",
        "It validates the shared P1 Laplace assembly on analytic problems, checks",
        "Dirichlet markers and real-scale geometry, tests field sign and symmetry,",
        "compares raw element fields with recovered-gradient minima, and reruns the",
        "coherent E1-relative E2/E3-swapped reference benchmark under refinement.",
        "Failed rows and flagged candidates remain in the CSV outputs.",
        "",
        "## Analytic validation",
        "",
        "| problem | quantity | value | tolerance | pass | samples | units |",
        "|:---|:---|---:|---:|:---:|---:|:---|",
    ]
    for metric in study.concentric.metrics + study.uniform.metrics + study.symmetry.metrics:
        lines.append(
            f"| {metric.problem} | {metric.quantity} | {metric.value:.8g} "
            f"| {metric.tolerance:.8g} | {_yes_no(metric.passed)} "
            f"| {metric.sample_count} | {metric.units} |"
        )
    lines.extend(
        [
            "",
            f"Analytic Laplace and sign tests pass: **{_yes_no(analytic_pass)}**.",
            f"Undisplaced four-electrode symmetry tests at h={study.symmetry.mesh_size_m * 1e3:.6g} mm pass: **{_yes_no(symmetry_pass)}**.",
            "",
            "## Boundary markers",
            "",
            "| boundary | nodes | expected V | geometry residual (m) | potential error (V) | overlaps | missing | complete |",
            "|:---|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for item in study.boundary_markers:
        lines.append(
            f"| {item.boundary} | {item.node_count} | {item.expected_potential_v:.6g} "
            f"| {item.maximum_geometry_residual_m:.6g} "
            f"| {item.maximum_potential_error_v:.6g} | {item.overlap_node_count} "
            f"| {item.missing_boundary_node_count} | {_yes_no(item.complete)} |"
        )
    minimum_gap = min(item.minimum_electrode_gap_m for item in study.geometry_cases)
    minimum_outer = min(item.minimum_outer_clearance_m for item in study.geometry_cases)
    lines.extend(
        [
            "",
            f"All boundary markers are complete and exclusive: **{_yes_no(boundary_pass)}**.",
            "",
            "## Geometry",
            "",
            "Gmsh constructs the vacuum as an exact OpenCASCADE outer disk minus",
            "four exact circular disks; the P1 mesh represents each curve by chords.",
            f"The requested radius is {REAL_ELECTRODE_RADIUS_M * 1e3:.6g} mm, the outer",
            f"radius is {REAL_OUTER_BOUNDARY_RADIUS_M * 1e3:.6g} mm, and the centre",
            f"radius is {electrode_center_radius_m(REAL_INNER_RADIUS_M, REAL_ELECTRODE_RADIUS_M) * 1e3:.6g} mm.",
            f"Across nominal geometry and reference rows 1--10, the minimum electrode",
            f"gap is {minimum_gap * 1e3:.6g} mm and minimum outer clearance is",
            f"{minimum_outer * 1e3:.6g} mm. All cases are valid: **{_yes_no(geometry_pass)}**.",
            "",
            "## Recovered-gradient candidate audit",
            "",
            f"Artifact action: `{study.artifact_action}`. The default audit action only",
            "flags; it never alters forward outputs. The documented flag requires a",
            "candidate within 0.02 mesh lengths of a facet together with an adjacent",
            "raw-field jump ratio of at least 0.50, or recovered psi at least 100 times",
            "the lowest validated candidate in that row.",
            f"Flagged candidates: `{flagged}/{candidate_records}`.",
            f"Flagged selected minima: `{flagged_selected}/{selected_records}` "
            f"(`{flagged_selected_fraction * 100.0:.3f}%`).",
            "A flagged-selected fraction above 10% is conservatively treated as",
            "numerically material for the scientific conclusion.",
            "Raw P1 fields are constant per triangle and generally do not vanish at a",
            "recovered-field zero; that disagreement is reported but is not alone an",
            "artifact criterion.",
            "",
            "## Reference-validation refinement",
            "",
            "| h (mm) | rows | completed | exact-three | mean (mm) | median (mm) | max (mm) | p95 (mm) | runtime (s) |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for case in sorted(study.refinement_cases, key=lambda item: item.mesh_size_m, reverse=True):
        summary = case.report.summary()
        lines.append(
            f"| {case.mesh_size_m * 1e3:.6g} | {summary.selected_rows} "
            f"| {summary.completed_rows} | {summary.rows_with_exactly_three_physical_minima} "
            f"| {_mm(summary.mean_error_m)} | {_mm(summary.median_error_m)} "
            f"| {_mm(summary.maximum_error_m)} | {_mm(summary.percentile_95_error_m)} "
            f"| {case.report.runtime_seconds:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Comparable rows 1--{assessment.selected_rows} mean-error change from",
            f"h={assessment.coarse_mesh_size_m * 1e3:.6g} mm to",
            f"h={assessment.fine_mesh_size_m * 1e3:.6g} mm is",
            f"`{assessment.relative_error_reduction * 100.0:.3f}%` reduction.",
            f"At least 5% improvement: **{_yes_no(assessment.meaningful_error_reduction)}**.",
            f"Exactly-three topology reaches `{assessment.fine_exactly_three_rows}/{assessment.selected_rows}` and is stable for every row:",
            f"**{_yes_no(assessment.topology_stable)}**.",
            *partial_refinement_lines,
            "",
            "## Numerical-bug and scientific conclusion",
            "",
            (
                "No FEM assembly, sign, boundary-marker, or geometry bug was found."
                if core_numerics_pass
                else "At least one FEM analytic, symmetry, boundary, or geometry audit failed; model-class conclusions must remain suspended."
            ),
            (
                "The remaining mismatch is scientifically consistent with a model-class/topology limitation: core numerical audits pass, refinement does not reduce error by the documented 5% threshold, and artifact flags do not dominate selected minima."
                if model_class_conclusion_justified
                else "It is not yet scientifically justified to call the remaining mismatch model-class limited under the documented numerical criteria."
            ),
            f"Model-class conclusion justified: **{_yes_no(model_class_conclusion_justified)}**.",
            "No ML or synthetic dataset generation was performed.",
            "",
            f"Total audit runtime: `{study.runtime_seconds:.3f} s`.",
            "",
        ]
    )
    return "\n".join(lines)


def _plot_concentric_potential(result: ConcentricAuditResult, path: Path) -> None:
    radius = np.linalg.norm(result.mesh.p.T, axis=1) * 1.0e3
    order = np.argsort(radius)
    figure = Figure(figsize=(7.4, 5.0), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.scatter(radius, result.numerical_potential_v, s=8, alpha=0.35, label="P1 FEM nodes")
    axis.plot(radius[order], result.analytic_potential_v[order], color="C1", linewidth=2, label="analytic")
    axis.set(xlabel="radius (mm)", ylabel="potential (V)", title="Concentric capacitor potential")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=180)


def _plot_concentric_field(result: ConcentricAuditResult, path: Path) -> None:
    radius = np.linalg.norm(result.mesh.p.T, axis=1)[result.field_mask] * 1.0e3
    numerical = np.linalg.norm(result.numerical_field_v_per_m[result.field_mask], axis=1)
    analytic = np.linalg.norm(result.analytic_field_v_per_m[result.field_mask], axis=1)
    order = np.argsort(radius)
    figure = Figure(figsize=(7.4, 5.0), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.scatter(radius, numerical, s=8, alpha=0.35, label="recovered FEM")
    axis.plot(radius[order], analytic[order], color="C1", linewidth=2, label="analytic")
    axis.set(xlabel="radius (mm)", ylabel="|E| (V/m)", title="Concentric capacitor field magnitude")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=180)


def _plot_uniform_field(result: UniformFieldAuditResult, path: Path) -> None:
    errors = result.numerical_field_v_per_m - np.asarray((-1.0, 0.0))
    figure = Figure(figsize=(6.5, 5.2), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    image = axis.scatter(
        result.mesh.p[0],
        result.mesh.p[1],
        c=np.linalg.norm(errors, axis=1),
        s=15,
        cmap="viridis",
    )
    figure.colorbar(image, ax=axis, label="|E - (-1,0)| (V/m)")
    axis.set_aspect("equal")
    axis.set(xlabel="x (m)", ylabel="y (m)", title="Uniform-field sign and recovery error")
    figure.savefig(path, dpi=180)


def _plot_refinement_error(cases: Sequence[MeshRefinementCase], path: Path) -> None:
    full_count = max(len(case.row_numbers) for case in cases)
    comparable = sorted(
        (case for case in cases if len(case.row_numbers) == full_count),
        key=lambda item: item.mesh_size_m,
        reverse=True,
    )
    mesh_mm = [case.mesh_size_m * 1.0e3 for case in comparable]
    mean_mm = [case.report.summary().mean_error_m * 1.0e3 for case in comparable]
    maximum_mm = [case.report.summary().maximum_error_m * 1.0e3 for case in comparable]
    figure = Figure(figsize=(7.2, 4.8), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.plot(mesh_mm, mean_mm, marker="o", label="mean")
    axis.plot(mesh_mm, maximum_mm, marker="s", label="maximum")
    axis.invert_xaxis()
    axis.set(xlabel="mesh size h (mm; finer to right)", ylabel="assignment error (mm)", title="Reference error under refinement")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=180)


def _plot_refinement_topology(cases: Sequence[MeshRefinementCase], path: Path) -> None:
    full_count = max(len(case.row_numbers) for case in cases)
    comparable = sorted(
        (case for case in cases if len(case.row_numbers) == full_count),
        key=lambda item: item.mesh_size_m,
        reverse=True,
    )
    mesh_mm = [case.mesh_size_m * 1.0e3 for case in comparable]
    exact = [case.report.summary().rows_with_exactly_three_physical_minima for case in comparable]
    figure = Figure(figsize=(7.2, 4.8), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.plot(mesh_mm, exact, marker="o", color="C2")
    axis.axhline(full_count, color="0.4", linestyle="--", label="all rows")
    axis.invert_xaxis()
    axis.set(xlabel="mesh size h (mm; finer to right)", ylabel="rows with exactly three candidates", title="Topology under refinement")
    axis.set_ylim(0, full_count + 0.5)
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=180)


def _plot_artifacts(records: Sequence[dict[str, object]], path: Path) -> None:
    usable = [record for record in records if record.get("status") == "ok"]
    figure = Figure(figsize=(7.2, 5.0), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    colors = ["C3" if record.get("artifact_flagged") else "C0" for record in usable]
    axis.scatter(
        [float(record["nearest_facet_distance_over_mesh_h"]) for record in usable],
        [float(record["adjacent_raw_field_jump_ratio"]) for record in usable],
        c=colors,
        alpha=0.8,
    )
    axis.axvline(0.02, color="0.4", linestyle="--", label="facet threshold")
    axis.axhline(0.50, color="0.6", linestyle=":", label="jump threshold")
    axis.set(xlabel="nearest facet distance / mesh h", ylabel="adjacent raw-field jump ratio", title="Recovered-gradient candidate artifact flags")
    axis.set_xscale("log")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=180)


def _mm(value_m: float) -> str:
    return "n/a" if not np.isfinite(value_m) else f"{value_m * 1.0e3:.6g}"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _parse_mesh_sizes_mm(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) * 1.0e-3 for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("mesh sizes must be comma-separated numbers") from error
    if not values or any(not np.isfinite(item) or item <= 0.0 for item in values):
        raise argparse.ArgumentTypeError("mesh sizes must be finite and positive")
    return values


def build_parser() -> argparse.ArgumentParser:
    """Build the Milestone-7 FEM numerical-audit command line."""

    parser = argparse.ArgumentParser(
        prog="rf-trap-fem-audit",
        description="Audit analytic FEM accuracy, markers, geometry, and refinement.",
    )
    parser.add_argument("input", nargs="?", type=Path, default=Path("Data.txt"))
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("validation_results") / "milestone_7",
    )
    parser.add_argument(
        "--refinement-mesh-sizes-mm",
        type=_parse_mesh_sizes_mm,
        default=(2.0e-3, 1.5e-3, 1.0e-3, 0.75e-3),
    )
    parser.add_argument("--skip-half-mm", action="store_true")
    parser.add_argument("--artifact-action", choices=("flag", "filter"), default="flag")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run Milestone 7, write artifacts, and print the scientific gate summary."""

    arguments = build_parser().parse_args(argv)
    dataset = load_reference_dataset(arguments.input)
    study = run_fem_numerical_audit(
        dataset,
        refinement_mesh_sizes_m=arguments.refinement_mesh_sizes_mm,
        include_half_mm=not arguments.skip_half_mm,
        artifact_action=arguments.artifact_action,
    )
    paths = write_fem_numerical_audit_outputs(
        study,
        arguments.output_directory,
    )
    analytic_pass = all(
        metric.passed
        for metric in study.concentric.metrics + study.uniform.metrics
    )
    print(f"analytic FEM and sign tests pass: {_yes_no(analytic_pass)}")
    print(
        "boundary markers complete: "
        + _yes_no(all(item.complete for item in study.boundary_markers))
    )
    print(
        "geometry cases valid: "
        + _yes_no(all(item.valid for item in study.geometry_cases))
    )
    print(
        "meaningful reference error reduction: "
        + _yes_no(study.refinement_assessment.meaningful_error_reduction)
    )
    print(
        "fine-mesh topology stable: "
        + _yes_no(study.refinement_assessment.topology_stable)
    )
    print(f"runtime: {study.runtime_seconds:.3f} s")
    print(f"report: {paths.markdown_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
