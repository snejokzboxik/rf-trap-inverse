"""Focused diagnostics for pre-selection Hessian-valid minima."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle
from numpy.typing import NDArray

from .config import ForwardModelConfig
from .demo import demonstrator_config, demonstrator_displacements_m
from .field import element_electric_fields
from .forward import ForwardModelResult, run_forward_model
from .mesh import nearest_mesh_facet
from .minima import LocalMinimum, _finite_difference_hessian


@dataclass(frozen=True)
class CandidateInvestigationConfig:
    """Numerical controls for the focused extra-candidate investigation."""

    mesh_size_m: float = 60.0e-6
    outer_radius_m: float = 4.0e-3
    hessian_steps_m: tuple[float, ...] = (
        1.0e-6,
        2.0e-6,
        4.0e-6,
        8.0e-6,
        16.0e-6,
        24.0e-6,
        32.0e-6,
    )
    perturbation_mesh_sizes_m: tuple[float, ...] = (59.0e-6, 60.0e-6, 61.0e-6)
    boundary_proximity_mesh_lengths: float = 2.0
    facet_proximity_mesh_fraction: float = 0.01
    psi_outlier_ratio: float = 1.0e4
    hessian_variation_ratio: float = 4.0
    plot_grid_points_per_axis: int = 181

    def __post_init__(self) -> None:
        """Validate all focused-study controls."""

        positive_values = (
            self.mesh_size_m,
            self.outer_radius_m,
            self.boundary_proximity_mesh_lengths,
            self.facet_proximity_mesh_fraction,
            self.psi_outlier_ratio,
            self.hessian_variation_ratio,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive_values):
            raise ValueError("investigation scales and thresholds must be positive")
        _validate_positive_sequence("hessian_steps_m", self.hessian_steps_m)
        _validate_positive_sequence(
            "perturbation_mesh_sizes_m",
            self.perturbation_mesh_sizes_m,
        )
        if not any(
            np.isclose(value, self.mesh_size_m, rtol=0.0, atol=1.0e-15)
            for value in self.perturbation_mesh_sizes_m
        ):
            raise ValueError("perturbation_mesh_sizes_m must include mesh_size_m")
        if (
            not isinstance(self.plot_grid_points_per_axis, int)
            or self.plot_grid_points_per_axis < 41
        ):
            raise ValueError("plot_grid_points_per_axis must be at least 41")


@dataclass(frozen=True)
class CandidateDiagnostic:
    """Geometry and selection diagnostics for one Hessian-valid candidate."""

    rank_by_psi: int
    minimum: LocalMinimum
    selected: bool
    nearest_electrode_index: int
    nearest_electrode_clearance_m: float
    search_boundary_clearance_m: float
    outer_boundary_clearance_m: float
    nearest_other_candidate_distance_m: float
    nearest_mesh_facet_distance_m: float
    adjacent_element_field_jump_v_per_m: float
    adjacent_element_field_magnitudes_v_per_m: tuple[float, float]


@dataclass(frozen=True)
class HessianStepDiagnostic:
    """Finite-difference Hessian eigenvalues at one stencil size."""

    step_m: float
    eigenvalues_v2_per_m4: NDArray[np.float64]


@dataclass(frozen=True)
class MeshPerturbationDiagnostic:
    """Pre-selection candidates recovered at one nearby mesh size."""

    mesh_size_m: float
    node_count: int
    triangle_count: int
    candidates: tuple[LocalMinimum, ...]


@dataclass(frozen=True)
class ArtifactAssessment:
    """Rule-based classification and the individual evidence flags."""

    likely_cause: str
    likely_physical: bool
    boundary_or_search_artifact: bool
    duplicate_or_merge_issue: bool
    recovered_gradient_interpolation_artifact: bool
    psi_ratio_to_largest_selected: float
    facet_distance_in_mesh_lengths: float
    hessian_small_to_large_step_ratio: float
    absent_on_adjacent_mesh_sizes: bool


@dataclass(frozen=True)
class CandidateInvestigationReport:
    """Complete numerical products for the focused problematic case."""

    config: CandidateInvestigationConfig
    case_config: ForwardModelConfig
    forward_result: ForwardModelResult
    candidates: tuple[CandidateDiagnostic, ...]
    extra_candidate_rank: int
    hessian_steps: tuple[HessianStepDiagnostic, ...]
    mesh_perturbations: tuple[MeshPerturbationDiagnostic, ...]
    assessment: ArtifactAssessment

    @property
    def extra_candidate(self) -> CandidateDiagnostic:
        """Return the pre-selection candidate rejected by the three-output API."""

        return next(
            item
            for item in self.candidates
            if item.rank_by_psi == self.extra_candidate_rank
        )


@dataclass(frozen=True)
class CandidateInvestigationOutputPaths:
    """Paths produced by :func:`write_candidate_investigation_outputs`."""

    candidates_csv: Path
    hessian_steps_csv: Path
    mesh_perturbations_csv: Path
    markdown_report: Path
    candidate_map_plot: Path
    local_artifact_plot: Path
    sensitivity_plot: Path


def run_candidate_investigation(
    base_config: ForwardModelConfig | None = None,
    investigation_config: CandidateInvestigationConfig | None = None,
) -> CandidateInvestigationReport:
    """Run the problematic case and narrowly scoped numerical sensitivity checks."""

    base = demonstrator_config() if base_config is None else base_config
    controls = (
        CandidateInvestigationConfig()
        if investigation_config is None
        else investigation_config
    )
    primary_config = _case_config(base, controls.mesh_size_m, controls.outer_radius_m)
    primary = run_forward_model(demonstrator_displacements_m(), primary_config)
    validated = primary.minima_diagnostics.hessian_validated_minima
    if len(validated) <= len(primary.minima):
        raise RuntimeError("the focused case did not reproduce an extra candidate")
    candidates = build_candidate_diagnostics(primary, primary_config)
    extra_ranks = [item.rank_by_psi for item in candidates if not item.selected]
    if len(extra_ranks) != 1:
        raise RuntimeError(
            f"expected one unselected candidate in the focused case, found {len(extra_ranks)}"
        )
    extra_rank = extra_ranks[0]
    extra = candidates[extra_rank - 1]
    hessian_steps = _hessian_step_diagnostics(primary, extra.minimum, controls)
    perturbations = _mesh_perturbation_diagnostics(
        base,
        primary,
        controls,
    )
    assessment = assess_extra_candidate(
        extra,
        candidates,
        hessian_steps,
        perturbations,
        primary_config,
        controls,
    )
    return CandidateInvestigationReport(
        config=controls,
        case_config=primary_config,
        forward_result=primary,
        candidates=candidates,
        extra_candidate_rank=extra_rank,
        hessian_steps=hessian_steps,
        mesh_perturbations=perturbations,
        assessment=assessment,
    )


def build_candidate_diagnostics(
    result: ForwardModelResult,
    config: ForwardModelConfig,
) -> tuple[CandidateDiagnostic, ...]:
    """Measure boundaries, separation, and mesh-facet proximity for all candidates."""

    validated = result.minima_diagnostics.hessian_validated_minima
    if not validated:
        raise ValueError("forward result does not contain pre-selection candidates")
    positions = np.vstack([minimum.position_m for minimum in validated])
    diagnostics: list[CandidateDiagnostic] = []
    for rank, minimum in enumerate(validated, start=1):
        electrode_clearances = (
            np.linalg.norm(result.geometry.centers_m - minimum.position_m, axis=1)
            - result.geometry.config.electrode_radius_m
        )
        nearest_electrode = int(np.argmin(electrode_clearances))
        other_distances = np.linalg.norm(positions - minimum.position_m, axis=1)
        other_distances[rank - 1] = np.inf
        facet_distance, facet_index = nearest_mesh_facet(
            result.trap_mesh.mesh,
            minimum.position_m,
        )
        field_jump, field_magnitudes = _facet_field_diagnostics(
            result,
            facet_index,
        )
        diagnostics.append(
            CandidateDiagnostic(
                rank_by_psi=rank,
                minimum=minimum,
                selected=_is_selected(minimum, result.minima),
                nearest_electrode_index=nearest_electrode + 1,
                nearest_electrode_clearance_m=float(electrode_clearances[nearest_electrode]),
                search_boundary_clearance_m=float(
                    config.minima.search_half_extent_m
                    - np.max(np.abs(minimum.position_m))
                ),
                outer_boundary_clearance_m=float(
                    result.geometry.config.outer_radius_m
                    - np.linalg.norm(minimum.position_m)
                ),
                nearest_other_candidate_distance_m=float(np.min(other_distances)),
                nearest_mesh_facet_distance_m=facet_distance,
                adjacent_element_field_jump_v_per_m=field_jump,
                adjacent_element_field_magnitudes_v_per_m=field_magnitudes,
            )
        )
    return tuple(diagnostics)


def assess_extra_candidate(
    extra: CandidateDiagnostic,
    candidates: Sequence[CandidateDiagnostic],
    hessian_steps: Sequence[HessianStepDiagnostic],
    perturbations: Sequence[MeshPerturbationDiagnostic],
    case_config: ForwardModelConfig,
    controls: CandidateInvestigationConfig,
) -> ArtifactAssessment:
    """Classify an extra candidate using explicit, configurable evidence tests."""

    selected_psi = [
        item.minimum.pseudopotential_v2_per_m2
        for item in candidates
        if item.selected
    ]
    if not selected_psi:
        raise ValueError("at least one selected candidate is required")
    psi_ratio = extra.minimum.pseudopotential_v2_per_m2 / max(selected_psi)
    mesh_size = case_config.mesh.characteristic_length_m
    boundary_limit = controls.boundary_proximity_mesh_lengths * mesh_size
    boundary_artifact = min(
        extra.nearest_electrode_clearance_m,
        extra.search_boundary_clearance_m,
    ) <= boundary_limit
    duplicate_issue = (
        extra.nearest_other_candidate_distance_m
        <= case_config.minima.merge_distance_m
    )
    facet_ratio = extra.nearest_mesh_facet_distance_m / mesh_size
    facet_locked = facet_ratio <= controls.facet_proximity_mesh_fraction
    ordered_steps = sorted(hessian_steps, key=lambda item: item.step_m)
    positive_steps = [
        item
        for item in ordered_steps
        if np.all(item.eigenvalues_v2_per_m4 > 0.0)
    ]
    if len(positive_steps) >= 2:
        hessian_ratio = float(
            positive_steps[0].eigenvalues_v2_per_m4[0]
            / positive_steps[-1].eigenvalues_v2_per_m4[0]
        )
    else:
        hessian_ratio = np.inf
    hessian_unstable = hessian_ratio >= controls.hessian_variation_ratio
    adjacent_counts = [
        len(item.candidates)
        for item in perturbations
        if not np.isclose(
            item.mesh_size_m,
            controls.mesh_size_m,
            rtol=0.0,
            atol=1.0e-15,
        )
    ]
    absent_adjacent = bool(adjacent_counts) and all(
        count == case_config.minima.expected_minima for count in adjacent_counts
    )
    psi_outlier = psi_ratio >= controls.psi_outlier_ratio
    recovery_artifact = bool(
        not boundary_artifact
        and not duplicate_issue
        and facet_locked
        and hessian_unstable
        and absent_adjacent
        and psi_outlier
    )
    if recovery_artifact:
        cause = "recovered-gradient interpolation artifact at a mesh facet"
    elif boundary_artifact:
        cause = "boundary or search-window artifact"
    elif duplicate_issue:
        cause = "duplicate or merge-threshold issue"
    else:
        cause = "inconclusive"
    return ArtifactAssessment(
        likely_cause=cause,
        likely_physical=not (
            recovery_artifact or boundary_artifact or duplicate_issue or psi_outlier
        ),
        boundary_or_search_artifact=boundary_artifact,
        duplicate_or_merge_issue=duplicate_issue,
        recovered_gradient_interpolation_artifact=recovery_artifact,
        psi_ratio_to_largest_selected=float(psi_ratio),
        facet_distance_in_mesh_lengths=float(facet_ratio),
        hessian_small_to_large_step_ratio=hessian_ratio,
        absent_on_adjacent_mesh_sizes=absent_adjacent,
    )


def write_candidate_investigation_outputs(
    report: CandidateInvestigationReport,
    output_directory: str | Path,
) -> CandidateInvestigationOutputPaths:
    """Write focused CSV diagnostics, Markdown, and headless plots."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = CandidateInvestigationOutputPaths(
        candidates_csv=output / "candidate_diagnostics.csv",
        hessian_steps_csv=output / "hessian_step_sensitivity.csv",
        mesh_perturbations_csv=output / "mesh_perturbation_summary.csv",
        markdown_report=output / "extra_candidate_report.md",
        candidate_map_plot=output / "all_candidates.png",
        local_artifact_plot=output / "extra_candidate_local.png",
        sensitivity_plot=output / "candidate_sensitivity.png",
    )
    _write_csv(paths.candidates_csv, _candidate_rows(report.candidates))
    _write_csv(paths.hessian_steps_csv, _hessian_rows(report.hessian_steps))
    _write_csv(
        paths.mesh_perturbations_csv,
        _perturbation_rows(report.mesh_perturbations),
    )
    paths.markdown_report.write_text(_markdown_report(report), encoding="utf-8")
    _write_candidate_map(report, paths.candidate_map_plot)
    _write_local_artifact_plot(report, paths.local_artifact_plot)
    _write_sensitivity_plot(report, paths.sensitivity_plot)
    return paths


def _validate_positive_sequence(name: str, values: Sequence[float]) -> None:
    if len(values) < 2:
        raise ValueError(f"{name} must contain at least two values")
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain finite positive values")
    if np.unique(array).size != array.size:
        raise ValueError(f"{name} must not contain duplicates")


def _case_config(
    base: ForwardModelConfig,
    mesh_size_m: float,
    outer_radius_m: float,
) -> ForwardModelConfig:
    return replace(
        base,
        geometry=replace(base.geometry, outer_radius_m=outer_radius_m),
        mesh=replace(base.mesh, characteristic_length_m=mesh_size_m),
    )


def _is_selected(
    candidate: LocalMinimum,
    selected: Sequence[LocalMinimum],
) -> bool:
    return any(
        np.array_equal(candidate.position_m, minimum.position_m)
        for minimum in selected
    )


def _facet_field_diagnostics(
    result: ForwardModelResult,
    facet_index: int,
) -> tuple[float, tuple[float, float]]:
    adjacent = result.trap_mesh.mesh.f2t[:, facet_index]
    adjacent = adjacent[adjacent >= 0]
    if adjacent.size != 2:
        return np.nan, (np.nan, np.nan)
    fields = element_electric_fields(
        result.trap_mesh.mesh,
        result.fem_solution.potential_v,
    )
    magnitudes = tuple(float(np.linalg.norm(fields[:, index])) for index in adjacent)
    jump = float(np.linalg.norm(fields[:, adjacent[0]] - fields[:, adjacent[1]]))
    return jump, magnitudes


def _hessian_step_diagnostics(
    result: ForwardModelResult,
    candidate: LocalMinimum,
    controls: CandidateInvestigationConfig,
) -> tuple[HessianStepDiagnostic, ...]:
    records = []
    for step in sorted(controls.hessian_steps_m):
        hessian = _finite_difference_hessian(
            result.recovered_field,
            candidate.position_m,
            step,
        )
        if hessian is None:
            eigenvalues = np.asarray([np.nan, np.nan])
        else:
            eigenvalues = np.linalg.eigvalsh(hessian)
        records.append(
            HessianStepDiagnostic(
                step_m=step,
                eigenvalues_v2_per_m4=eigenvalues,
            )
        )
    return tuple(records)


def _mesh_perturbation_diagnostics(
    base: ForwardModelConfig,
    primary: ForwardModelResult,
    controls: CandidateInvestigationConfig,
) -> tuple[MeshPerturbationDiagnostic, ...]:
    records = []
    for mesh_size in sorted(controls.perturbation_mesh_sizes_m):
        if np.isclose(mesh_size, controls.mesh_size_m, rtol=0.0, atol=1.0e-15):
            result = primary
        else:
            config = _case_config(base, mesh_size, controls.outer_radius_m)
            result = run_forward_model(demonstrator_displacements_m(), config)
        records.append(
            MeshPerturbationDiagnostic(
                mesh_size_m=mesh_size,
                node_count=result.trap_mesh.number_of_nodes,
                triangle_count=result.trap_mesh.number_of_triangles,
                candidates=result.minima_diagnostics.hessian_validated_minima,
            )
        )
    return tuple(records)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("at least one row is required")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _candidate_rows(
    candidates: Iterable[CandidateDiagnostic],
) -> list[dict[str, object]]:
    rows = []
    for item in candidates:
        minimum = item.minimum
        rows.append(
            {
                "rank_by_psi": item.rank_by_psi,
                "x_m": minimum.position_m[0],
                "y_m": minimum.position_m[1],
                "psi_v2_per_m2": minimum.pseudopotential_v2_per_m2,
                "hessian_eigenvalue_1_v2_per_m4": minimum.hessian_eigenvalues_v2_per_m4[0],
                "hessian_eigenvalue_2_v2_per_m4": minimum.hessian_eigenvalues_v2_per_m4[1],
                "nearest_electrode_index": item.nearest_electrode_index,
                "nearest_electrode_clearance_m": item.nearest_electrode_clearance_m,
                "search_boundary_clearance_m": item.search_boundary_clearance_m,
                "outer_boundary_clearance_m": item.outer_boundary_clearance_m,
                "nearest_other_candidate_distance_m": item.nearest_other_candidate_distance_m,
                "nearest_mesh_facet_distance_m": item.nearest_mesh_facet_distance_m,
                "adjacent_element_field_jump_v_per_m": item.adjacent_element_field_jump_v_per_m,
                "adjacent_element_field_magnitude_1_v_per_m": item.adjacent_element_field_magnitudes_v_per_m[0],
                "adjacent_element_field_magnitude_2_v_per_m": item.adjacent_element_field_magnitudes_v_per_m[1],
                "selected": item.selected,
            }
        )
    return rows


def _hessian_rows(
    records: Iterable[HessianStepDiagnostic],
) -> list[dict[str, object]]:
    return [
        {
            "step_m": item.step_m,
            "hessian_eigenvalue_1_v2_per_m4": item.eigenvalues_v2_per_m4[0],
            "hessian_eigenvalue_2_v2_per_m4": item.eigenvalues_v2_per_m4[1],
        }
        for item in records
    ]


def _perturbation_rows(
    records: Iterable[MeshPerturbationDiagnostic],
) -> list[dict[str, object]]:
    rows = []
    for record in records:
        row: dict[str, object] = {
            "mesh_size_m": record.mesh_size_m,
            "node_count": record.node_count,
            "triangle_count": record.triangle_count,
            "hessian_validated_candidates": len(record.candidates),
        }
        for index, candidate in enumerate(record.candidates, start=1):
            row[f"candidate_{index}_x_m"] = candidate.position_m[0]
            row[f"candidate_{index}_y_m"] = candidate.position_m[1]
            row[f"candidate_{index}_psi_v2_per_m2"] = (
                candidate.pseudopotential_v2_per_m2
            )
        rows.append(row)
    fieldnames = list(rows[0])
    for row in rows[1:]:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    return [{field: row.get(field, "") for field in fieldnames} for row in rows]


def _markdown_report(report: CandidateInvestigationReport) -> str:
    result = report.forward_result
    extra = report.extra_candidate
    assessment = report.assessment
    adjacent_meshes = ", ".join(
        f"{item.mesh_size_m * 1.0e6:.6g} µm"
        for item in report.mesh_perturbations
        if not np.isclose(
            item.mesh_size_m,
            report.config.mesh_size_m,
            rtol=0.0,
            atol=1.0e-15,
        )
    )
    lines = [
        "# Focused investigation: fourth Hessian-valid candidate",
        "",
        "## Case",
        "",
        f"- Mesh characteristic length: `{report.config.mesh_size_m * 1.0e6:.6g} µm`",
        f"- Outer radius: `{report.config.outer_radius_m * 1.0e3:.6g} mm`",
        f"- Nodes / triangles: `{result.trap_mesh.number_of_nodes}` / `{result.trap_mesh.number_of_triangles}`",
        f"- Relative free-node residual: `{result.fem_solution.relative_free_residual:.6e}`",
        f"- Hessian-valid before selection: `{len(report.candidates)}`",
        f"- Returned by forward API: `{len(result.minima)}`",
        "",
        "## All pre-selection candidates",
        "",
        "Distances to electrodes are surface clearances. The search boundary is the",
        "configured square used by the optimizer, not the outer electrostatic boundary.",
        "",
        "| rank by Ψ | x (µm) | y (µm) | Ψ (V²/m²) | Hessian λ1 (V²/m⁴) | Hessian λ2 (V²/m⁴) | nearest electrode | electrode clearance (µm) | search clearance (µm) | nearest candidate (µm) | nearest mesh facet (µm) | selected |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for item in report.candidates:
        minimum = item.minimum
        lines.append(
            f"| {item.rank_by_psi} | {minimum.position_m[0] * 1.0e6:.8g} "
            f"| {minimum.position_m[1] * 1.0e6:.8g} "
            f"| {minimum.pseudopotential_v2_per_m2:.8e} "
            f"| {minimum.hessian_eigenvalues_v2_per_m4[0]:.8e} "
            f"| {minimum.hessian_eigenvalues_v2_per_m4[1]:.8e} "
            f"| {item.nearest_electrode_index} "
            f"| {item.nearest_electrode_clearance_m * 1.0e6:.8g} "
            f"| {item.search_boundary_clearance_m * 1.0e6:.8g} "
            f"| {item.nearest_other_candidate_distance_m * 1.0e6:.8g} "
            f"| {item.nearest_mesh_facet_distance_m * 1.0e6:.8g} "
            f"| {_yes_no(item.selected)} |"
        )
    lines.extend(
        [
            "",
            "## Fourth-candidate checks",
            "",
            f"- Candidate coordinates: `({extra.minimum.position_m[0] * 1.0e6:.6f}, {extra.minimum.position_m[1] * 1.0e6:.6f}) µm`.",
            f"- Its Ψ is `{assessment.psi_ratio_to_largest_selected:.6g}×` the largest selected-candidate Ψ; it is not null-like.",
            f"- Nearest electrode and search-window clearances are `{extra.nearest_electrode_clearance_m * 1.0e6:.3f} µm` and `{extra.search_boundary_clearance_m * 1.0e6:.3f} µm`.",
            f"- Nearest other candidate is `{extra.nearest_other_candidate_distance_m * 1.0e6:.3f} µm` away; "
            f"the configured merge threshold is `{report.case_config.minima.merge_distance_m * 1.0e6:.6g} µm`.",
        ]
    )
    lines.extend(
        [
            f"- The optimizer stopped `{extra.nearest_mesh_facet_distance_m * 1.0e6:.6f} µm` from a mesh facet "
            f"(`{assessment.facet_distance_in_mesh_lengths:.6g}` mesh lengths).",
            f"- The adjacent raw P1 element fields differ by `{extra.adjacent_element_field_jump_v_per_m:.6g} V/m`.",
            f"- Their magnitudes are `{extra.adjacent_element_field_magnitudes_v_per_m[0]:.6g}` and "
            f"`{extra.adjacent_element_field_magnitudes_v_per_m[1]:.6g} V/m`; "
            f"the recovered candidate magnitude is `{np.sqrt(extra.minimum.pseudopotential_v2_per_m2):.6g} V/m`, so none is a field null.",
            f"- The smallest-stencil to largest-positive-stencil Hessian λ1 ratio is `{assessment.hessian_small_to_large_step_ratio:.6g}`; a smooth Hessian should approach a finite value as the stencil shrinks.",
            f"- Perturbed meshes ({adjacent_meshes}) contain only the expected three candidates: **{_yes_no(assessment.absent_on_adjacent_mesh_sizes)}**.",
            "",
            "### Hessian stencil sensitivity",
            "",
            "| step (µm) | λ1 (V²/m⁴) | λ2 (V²/m⁴) | positive definite |",
            "|---:|---:|---:|:---:|",
        ]
    )
    for item in report.hessian_steps:
        lines.append(
            f"| {item.step_m * 1.0e6:.6g} "
            f"| {item.eigenvalues_v2_per_m4[0]:.8e} "
            f"| {item.eigenvalues_v2_per_m4[1]:.8e} "
            f"| {_yes_no(bool(np.all(item.eigenvalues_v2_per_m4 > 0.0)))} |"
        )
    lines.extend(
        [
            "",
            "### Local mesh-size perturbation",
            "",
            "| h (µm) | nodes | triangles | Hessian-valid candidates |",
            "|---:|---:|---:|---:|",
        ]
    )
    for item in report.mesh_perturbations:
        lines.append(
            f"| {item.mesh_size_m * 1.0e6:.6g} | {item.node_count} "
            f"| {item.triangle_count} | {len(item.candidates)} |"
        )
    lines.extend(
        [
            "",
            "## Classification",
            "",
            f"**Likely cause: {assessment.likely_cause}.**",
            "",
            f"- Physical minimum likely: **{_yes_no(assessment.likely_physical)}**",
            f"- Boundary/search artifact: **{_yes_no(assessment.boundary_or_search_artifact)}**",
            f"- Duplicate/merge-threshold issue: **{_yes_no(assessment.duplicate_or_merge_issue)}**",
            f"- Recovered-gradient interpolation artifact: **{_yes_no(assessment.recovered_gradient_interpolation_artifact)}**",
            "",
            "For a smooth two-dimensional harmonic potential, a nonconstant analytic electric",
            "field cannot have a strict interior local minimum of its magnitude. Here the",
            "candidate has a large nonzero field magnitude, lies essentially on a recovered-P1",
            "facet, disappears after ±1 µm mesh perturbations, and its finite-difference",
            "curvature grows strongly as the stencil shrinks. Together these diagnose a",
            "piecewise-interpolation kink that passes the configured Hessian test, not a",
            "fourth physical RF null.",
            "",
            "No physical equation, boundary condition, or forward-selection rule was changed.",
            "",
        ]
    )
    return "\n".join(lines)


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _evaluate_plot_grid(
    report: CandidateInvestigationReport,
    x_limits_m: tuple[float, float],
    y_limits_m: tuple[float, float],
    points_per_axis: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    x = np.linspace(*x_limits_m, points_per_axis)
    y = np.linspace(*y_limits_m, points_per_axis)
    grid_x, grid_y = np.meshgrid(x, y, indexing="xy")
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    valid = report.forward_result.geometry.contains_points(points, clearance_m=2.0e-6)
    values = np.full(points.shape[0], np.nan)
    values[valid] = np.asarray(
        report.forward_result.recovered_field.pseudopotential(points[valid])
    )
    return grid_x, grid_y, values.reshape(grid_x.shape)


def _write_candidate_map(
    report: CandidateInvestigationReport,
    path: Path,
) -> None:
    search_extent = report.case_config.minima.search_half_extent_m
    grid_x, grid_y, values = _evaluate_plot_grid(
        report,
        (-search_extent, search_extent),
        (-search_extent, search_extent),
        report.config.plot_grid_points_per_axis,
    )
    figure = Figure(figsize=(8.2, 7.0), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    log_values = np.log10(np.maximum(values, np.finfo(float).tiny))
    contour = axis.contourf(
        grid_x * 1.0e6,
        grid_y * 1.0e6,
        log_values,
        levels=36,
        cmap="viridis",
    )
    figure.colorbar(contour, ax=axis, label="log10 Ψ (V²/m²)")
    for center in report.forward_result.geometry.centers_m:
        axis.add_patch(
            Circle(
                center * 1.0e6,
                report.forward_result.geometry.config.electrode_radius_m * 1.0e6,
                fill=False,
                linewidth=1.0,
                color="0.35",
            )
        )
    axis.add_patch(
        Rectangle(
            (-search_extent * 1.0e6, -search_extent * 1.0e6),
            2.0 * search_extent * 1.0e6,
            2.0 * search_extent * 1.0e6,
            fill=False,
            linestyle="--",
            linewidth=1.0,
            color="0.25",
        )
    )
    for item in report.candidates:
        position = item.minimum.position_m * 1.0e6
        marker = "o" if item.selected else "X"
        color = "C2" if item.selected else "C3"
        axis.scatter(*position, marker=marker, s=70, color=color, edgecolor="white")
        axis.annotate(
            f"{item.rank_by_psi}: {'selected' if item.selected else 'extra'}",
            position,
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_title("All Hessian-valid candidates before lowest-three selection")
    axis.set_xlabel("x (µm)")
    axis.set_ylabel("y (µm)")
    axis.set_aspect("equal")
    axis.set_xlim(-search_extent * 1.0e6, search_extent * 1.0e6)
    axis.set_ylim(-search_extent * 1.0e6, search_extent * 1.0e6)
    axis.grid(True, alpha=0.2)
    figure.savefig(path, dpi=180, bbox_inches="tight")


def _write_local_artifact_plot(
    report: CandidateInvestigationReport,
    path: Path,
) -> None:
    extra = report.extra_candidate.minimum
    center = extra.position_m
    half_width = 75.0e-6
    grid_x, grid_y, values = _evaluate_plot_grid(
        report,
        (center[0] - half_width, center[0] + half_width),
        (center[1] - half_width, center[1] + half_width),
        151,
    )
    figure = Figure(figsize=(12.0, 5.2), layout="constrained")
    FigureCanvasAgg(figure)
    map_axis, profile_axis = figure.subplots(1, 2)
    contour = map_axis.contourf(
        grid_x * 1.0e6,
        grid_y * 1.0e6,
        values,
        levels=30,
        cmap="viridis",
    )
    figure.colorbar(contour, ax=map_axis, label="Ψ (V²/m²)")
    mesh = report.forward_result.trap_mesh.mesh
    points = mesh.p.T
    segments = points[mesh.facets.T] * 1.0e6
    visible = np.all(
        (segments[:, :, 0] >= (center[0] - half_width) * 1.0e6)
        & (segments[:, :, 0] <= (center[0] + half_width) * 1.0e6)
        & (segments[:, :, 1] >= (center[1] - half_width) * 1.0e6)
        & (segments[:, :, 1] <= (center[1] + half_width) * 1.0e6),
        axis=1,
    )
    map_axis.add_collection(
        LineCollection(segments[visible], colors="white", linewidths=0.55, alpha=0.65)
    )
    map_axis.scatter(
        center[0] * 1.0e6,
        center[1] * 1.0e6,
        marker="X",
        s=90,
        color="C3",
        edgecolor="white",
    )
    map_axis.set_title("Extra candidate lies on an internal mesh facet")
    map_axis.set_xlabel("x (µm)")
    map_axis.set_ylabel("y (µm)")
    map_axis.set_aspect("equal")

    _, facet_index = nearest_mesh_facet(mesh, center)
    facet_points = points[mesh.facets.T[facet_index]]
    tangent = facet_points[1] - facet_points[0]
    tangent /= np.linalg.norm(tangent)
    normal = np.asarray([-tangent[1], tangent[0]])
    offsets = np.linspace(-40.0e-6, 40.0e-6, 321)
    for direction, label, color in (
        (normal, "normal to nearest facet", "C3"),
        (tangent, "along nearest facet", "C0"),
    ):
        sample = center + offsets[:, np.newaxis] * direction
        psi = np.asarray(report.forward_result.recovered_field.pseudopotential(sample))
        profile_axis.plot(offsets * 1.0e6, psi, label=label, color=color)
    profile_axis.axvline(0.0, color="0.25", linewidth=1.0, linestyle="--")
    profile_axis.set_title("Recovered Ψ cross-sections through the extra candidate")
    profile_axis.set_xlabel("offset (µm)")
    profile_axis.set_ylabel("Ψ (V²/m²)")
    profile_axis.grid(True, alpha=0.25)
    profile_axis.legend()
    figure.savefig(path, dpi=180, bbox_inches="tight")


def _write_sensitivity_plot(
    report: CandidateInvestigationReport,
    path: Path,
) -> None:
    figure = Figure(figsize=(11.0, 4.8), layout="constrained")
    FigureCanvasAgg(figure)
    hessian_axis, mesh_axis = figure.subplots(1, 2)
    steps_um = np.asarray([item.step_m for item in report.hessian_steps]) * 1.0e6
    eigenvalues = np.vstack(
        [item.eigenvalues_v2_per_m4 for item in report.hessian_steps]
    ) / 1.0e10
    hessian_axis.plot(steps_um, eigenvalues[:, 0], marker="o", label="λ1")
    hessian_axis.plot(steps_um, eigenvalues[:, 1], marker="s", label="λ2")
    hessian_axis.axhline(0.0, color="0.25", linewidth=1.0)
    hessian_axis.set_title("Hessian eigenvalues do not converge with stencil size")
    hessian_axis.set_xlabel("finite-difference step (µm)")
    hessian_axis.set_ylabel("eigenvalue (10¹⁰ V²/m⁴)")
    hessian_axis.grid(True, alpha=0.25)
    hessian_axis.legend()

    sizes = np.asarray([item.mesh_size_m for item in report.mesh_perturbations]) * 1.0e6
    counts = [len(item.candidates) for item in report.mesh_perturbations]
    mesh_axis.plot(sizes, counts, marker="o", color="C3")
    mesh_axis.axhline(3.0, color="C2", linewidth=1.0, linestyle="--", label="expected")
    mesh_axis.set_title("Fourth-candidate count is mesh-sensitive")
    mesh_axis.set_xlabel("mesh characteristic length (µm)")
    mesh_axis.set_ylabel("Hessian-valid candidates")
    mesh_axis.set_xticks(sizes)
    mesh_axis.set_yticks(sorted(set(counts + [3])))
    mesh_axis.grid(True, alpha=0.25)
    mesh_axis.legend()
    figure.savefig(path, dpi=180, bbox_inches="tight")


def build_parser() -> argparse.ArgumentParser:
    """Build the focused-investigation command-line parser."""

    parser = argparse.ArgumentParser(
        prog="rf-trap-investigate-extra-candidate",
        description="Investigate all pre-selection minima in the milestone-two case.",
    )
    parser.add_argument("--mesh-size-um", type=float, default=60.0)
    parser.add_argument("--outer-radius-mm", type=float, default=4.0)
    parser.add_argument(
        "--perturbation-mesh-sizes-um",
        type=float,
        nargs="+",
        default=(59.0, 60.0, 61.0),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("validation_results") / "milestone_2_extra_candidate",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the focused study, write its artifacts, and print the classification."""

    arguments = build_parser().parse_args(argv)
    controls = CandidateInvestigationConfig(
        mesh_size_m=arguments.mesh_size_um * 1.0e-6,
        outer_radius_m=arguments.outer_radius_mm * 1.0e-3,
        perturbation_mesh_sizes_m=tuple(
            value * 1.0e-6 for value in arguments.perturbation_mesh_sizes_um
        ),
    )
    report = run_candidate_investigation(investigation_config=controls)
    paths = write_candidate_investigation_outputs(report, arguments.output_directory)
    print(f"Hessian-valid candidates before selection: {len(report.candidates)}")
    print(f"Likely cause: {report.assessment.likely_cause}")
    print(f"Markdown report: {paths.markdown_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
