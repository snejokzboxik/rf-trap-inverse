"""Milestone-7 analytic and mocked numerical-audit tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from rf_trap_forward.geometry import build_geometry, geometry_sanity
from rf_trap_forward.numerical_audit import (
    MeshRefinementCase,
    assess_mesh_refinement,
    audit_boundary_markers,
    run_concentric_capacitor_audit,
    run_uniform_field_audit,
)
from rf_trap_forward.real_scale import (
    REAL_ELECTRODE_RADIUS_M,
    REAL_INNER_RADIUS_M,
    electrode_center_radius_m,
    real_scale_forward_config,
)
from rf_trap_forward.reference_validation import ReferenceValidationSummary


def test_concentric_capacitor_matches_analytic_solution_on_coarse_mesh() -> None:
    """A loose coarse-mesh annulus test must bound potential and field errors."""

    result = run_concentric_capacitor_audit(mesh_size_m=5.0e-3)
    values = {metric.quantity: metric.value for metric in result.metrics}
    assert values["potential-relative-l2"] < 0.02
    assert values["potential-maximum-absolute-error"] < 0.03
    assert values["recovered-field-relative-l2"] < 0.10
    assert values["raw-element-field-relative-l2"] < 0.10


def test_uniform_field_has_negative_potential_gradient_sign() -> None:
    """The audit must fail conceptually if E is implemented as +grad(phi)."""

    result = run_uniform_field_audit(refinements=3)
    np.testing.assert_allclose(
        result.numerical_field_v_per_m,
        np.tile((-1.0, 0.0), (result.mesh.p.shape[1], 1)),
        rtol=0.0,
        atol=1.0e-12,
    )
    assert all(metric.passed for metric in result.metrics)


def test_boundary_markers_are_complete_and_exclusive(
    geometry,
    trap_mesh,
    fem_solution,
) -> None:
    """Every mesh-boundary node must belong to exactly one Dirichlet marker."""

    diagnostics = audit_boundary_markers(geometry, trap_mesh, fem_solution)
    assert len(diagnostics) == 5
    assert all(item.node_count > 0 for item in diagnostics)
    assert all(item.overlap_node_count == 0 for item in diagnostics)
    assert all(item.missing_boundary_node_count == 0 for item in diagnostics)
    assert all(item.complete for item in diagnostics)


def test_real_scale_geometry_sanity_reports_requested_clearances() -> None:
    """Real-scale centres, gaps, and containment must be explicit and positive."""

    config = real_scale_forward_config()
    geometry = build_geometry(config.geometry, np.zeros(6))
    sanity = geometry_sanity(config.geometry, geometry.centers_m)
    expected_radius = electrode_center_radius_m(
        REAL_INNER_RADIUS_M,
        REAL_ELECTRODE_RADIUS_M,
    )
    np.testing.assert_allclose(
        np.linalg.norm(geometry.centers_m, axis=1),
        expected_radius,
    )
    assert sanity.valid
    assert sanity.minimum_electrode_gap_m > 0.0
    assert sanity.minimum_outer_clearance_m > 0.0


@dataclass(frozen=True)
class _MockReport:
    summary_value: ReferenceValidationSummary

    def summary(self) -> ReferenceValidationSummary:
        """Return the fixed synthetic summary used by refinement tests."""

        return self.summary_value


def _summary(mean_m: float, exact: int) -> ReferenceValidationSummary:
    return ReferenceValidationSummary(
        selected_rows=10,
        completed_rows=10,
        failed_rows=0,
        rows_with_exactly_three_physical_minima=exact,
        matched_minima=30,
        mean_error_m=mean_m,
        median_error_m=mean_m,
        maximum_error_m=2.0 * mean_m,
        percentile_95_error_m=1.5 * mean_m,
    )


def test_mesh_refinement_summary_logic_uses_comparable_rows() -> None:
    """Mocked summary logic must ignore optional small-row half-mm cases."""

    cases = (
        MeshRefinementCase(2.0e-3, tuple(range(1, 11)), _MockReport(_summary(1.0e-3, 5))),
        MeshRefinementCase(0.75e-3, tuple(range(1, 11)), _MockReport(_summary(0.90e-3, 10))),
        MeshRefinementCase(0.50e-3, (1, 2, 3), _MockReport(_summary(0.10e-3, 3))),
    )
    assessment = assess_mesh_refinement(cases)
    assert assessment.relative_error_reduction == pytest.approx(0.10)
    assert assessment.meaningful_error_reduction
    assert assessment.topology_improved_or_equal
    assert assessment.topology_stable
    assert assessment.fine_mesh_size_m == 0.75e-3
