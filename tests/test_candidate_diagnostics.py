"""Synthetic tests for focused extra-candidate diagnostics."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import rf_trap_forward.candidate_diagnostics as diagnostics_module
from rf_trap_forward.candidate_diagnostics import (
    ArtifactAssessment,
    CandidateDiagnostic,
    CandidateInvestigationConfig,
    CandidateInvestigationReport,
    HessianStepDiagnostic,
    MeshPerturbationDiagnostic,
    assess_extra_candidate,
    build_candidate_diagnostics,
    write_candidate_investigation_outputs,
)
from rf_trap_forward.demo import demonstrator_config
from rf_trap_forward.minima import LocalMinimum


def _minimum(x_m: float, y_m: float, psi: float) -> LocalMinimum:
    return LocalMinimum(
        position_m=np.asarray([x_m, y_m]),
        pseudopotential_v2_per_m2=psi,
        hessian_eigenvalues_v2_per_m4=np.asarray([1.0e10, 2.0e10]),
        optimizer_succeeded=True,
    )


def _candidate(
    rank: int,
    minimum: LocalMinimum,
    selected: bool,
    *,
    electrode_clearance_m: float = 500.0e-6,
    search_clearance_m: float = 400.0e-6,
    other_distance_m: float = 220.0e-6,
    facet_distance_m: float = 0.01e-6,
) -> CandidateDiagnostic:
    return CandidateDiagnostic(
        rank_by_psi=rank,
        minimum=minimum,
        selected=selected,
        nearest_electrode_index=1,
        nearest_electrode_clearance_m=electrode_clearance_m,
        search_boundary_clearance_m=search_clearance_m,
        outer_boundary_clearance_m=3.5e-3,
        nearest_other_candidate_distance_m=other_distance_m,
        nearest_mesh_facet_distance_m=facet_distance_m,
        adjacent_element_field_jump_v_per_m=0.1,
        adjacent_element_field_magnitudes_v_per_m=(14.2, 14.1),
    )


def _synthetic_evidence():
    minima = (
        _minimum(0.3e-3, 0.25e-3, 2.0e-4),
        _minimum(0.32e-3, -0.22e-3, 4.0e-4),
        _minimum(-0.47e-3, 0.10e-3, 8.0e-4),
        _minimum(0.26e-3, 0.0, 200.0),
    )
    candidates = tuple(
        _candidate(index, minimum, index <= 3)
        for index, minimum in enumerate(minima, start=1)
    )
    hessian_steps = (
        HessianStepDiagnostic(1.0e-6, np.asarray([8.0e10, 20.0e10])),
        HessianStepDiagnostic(8.0e-6, np.asarray([1.0e10, 3.0e10])),
    )
    perturbations = (
        MeshPerturbationDiagnostic(59.0e-6, 100, 180, minima[:3]),
        MeshPerturbationDiagnostic(60.0e-6, 95, 170, minima),
        MeshPerturbationDiagnostic(61.0e-6, 90, 160, minima[:3]),
    )
    return minima, candidates, hessian_steps, perturbations


def test_forward_result_exposes_every_preselection_candidate(forward_result) -> None:
    """Stored candidate count and payload must agree before lowest-three selection."""

    diagnostics = forward_result.minima_diagnostics
    assert len(diagnostics.hessian_validated_minima) == diagnostics.hessian_validated_candidates
    measured = build_candidate_diagnostics(forward_result, demonstrator_config())
    assert len(measured) == diagnostics.hessian_validated_candidates
    assert all(item.selected for item in measured)
    assert all(item.nearest_electrode_clearance_m > 0.0 for item in measured)


def test_assessment_identifies_facet_locked_mesh_fragile_outlier() -> None:
    """The classification must combine boundary, merge, mesh, and Hessian evidence."""

    _, candidates, hessian_steps, perturbations = _synthetic_evidence()
    controls = CandidateInvestigationConfig()
    case_config = demonstrator_config()
    case_config = diagnostics_module._case_config(
        case_config,
        controls.mesh_size_m,
        controls.outer_radius_m,
    )
    assessment = assess_extra_candidate(
        candidates[-1],
        candidates,
        hessian_steps,
        perturbations,
        case_config,
        controls,
    )
    assert assessment.recovered_gradient_interpolation_artifact
    assert not assessment.likely_physical
    assert not assessment.boundary_or_search_artifact
    assert not assessment.duplicate_or_merge_issue


def test_writer_serializes_all_candidates_with_mocked_plots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Focused CSV and Markdown output must retain the unselected candidate."""

    minima, candidates, hessian_steps, perturbations = _synthetic_evidence()
    assessment = ArtifactAssessment(
        likely_cause="recovered-gradient interpolation artifact at a mesh facet",
        likely_physical=False,
        boundary_or_search_artifact=False,
        duplicate_or_merge_issue=False,
        recovered_gradient_interpolation_artifact=True,
        psi_ratio_to_largest_selected=250000.0,
        facet_distance_in_mesh_lengths=1.0e-4,
        hessian_small_to_large_step_ratio=8.0,
        absent_on_adjacent_mesh_sizes=True,
    )
    base_config = demonstrator_config()
    controls = CandidateInvestigationConfig()
    report = CandidateInvestigationReport(
        config=controls,
        case_config=diagnostics_module._case_config(
            base_config,
            controls.mesh_size_m,
            controls.outer_radius_m,
        ),
        forward_result=SimpleNamespace(
            minima=minima[:3],
            trap_mesh=SimpleNamespace(number_of_nodes=95, number_of_triangles=170),
            fem_solution=SimpleNamespace(relative_free_residual=1.0e-14),
        ),
        candidates=candidates,
        extra_candidate_rank=4,
        hessian_steps=hessian_steps,
        mesh_perturbations=perturbations,
        assessment=assessment,
    )

    def fake_plot(_, path: Path) -> None:
        path.write_bytes(b"mock PNG")

    monkeypatch.setattr(diagnostics_module, "_write_candidate_map", fake_plot)
    monkeypatch.setattr(diagnostics_module, "_write_local_artifact_plot", fake_plot)
    monkeypatch.setattr(diagnostics_module, "_write_sensitivity_plot", fake_plot)
    paths = write_candidate_investigation_outputs(report, tmp_path)

    with paths.candidates_csv.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    markdown = paths.markdown_report.read_text(encoding="utf-8")
    assert len(rows) == 4
    assert rows[-1]["selected"] == "False"
    assert "Hessian-valid before selection: `4`" in markdown
    assert "Recovered-gradient interpolation artifact: **yes**" in markdown


def test_investigation_config_requires_primary_mesh_in_perturbations() -> None:
    """The local sensitivity set must include the focused mesh itself."""

    with pytest.raises(ValueError, match="must include"):
        CandidateInvestigationConfig(
            perturbation_mesh_sizes_m=(58.0e-6, 59.0e-6, 61.0e-6),
        )
