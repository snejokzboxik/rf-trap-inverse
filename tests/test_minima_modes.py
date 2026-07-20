"""Synthetic-field tests for explicit and robust minima modes."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from skfem import MeshTri

from rf_trap_forward.config import GeometryConfig, MinimaSearchConfig
from rf_trap_forward.field import RecoveredField
from rf_trap_forward.geometry import TrapGeometry
from rf_trap_forward.minima_modes import (
    CandidateSeed,
    MinimaModeResult,
    RobustMinimaConfig,
    classify_facet_sensitive,
    classify_hessian_stability,
    compute_candidate_quality,
    run_minima_mode,
    select_robust_candidates,
)


def _synthetic_radial_field() -> tuple[RecoveredField, np.ndarray]:
    coordinates = np.linspace(-1.0, 1.0, 9)
    mesh = MeshTri.init_tensor(coordinates, coordinates)
    geometry_config = GeometryConfig(
        electrode_radius_m=0.05,
        nominal_centers_m=((-2.0, 2.0), (2.0, 2.0), (-2.0, -2.0), (2.0, -2.0)),
        outer_radius_m=5.0,
    )
    geometry = TrapGeometry(
        config=geometry_config,
        centers_m=np.asarray(geometry_config.nominal_centers_m),
        displacements_m=np.zeros(6),
    )
    nodal_field = mesh.p.copy()
    recovered = RecoveredField(geometry, mesh, nodal_field)
    potential = -0.5 * np.sum(mesh.p**2, axis=0)
    return recovered, potential


def test_candidate_quality_metric_computation() -> None:
    recovered, potential = _synthetic_radial_field()
    metrics = compute_candidate_quality(
        candidate_id=1,
        seed=CandidateSeed(np.zeros(2), ("synthetic-zero",), 1, True),
        recovered_field=recovered,
        potential_v=potential,
        candidate_psi_scale=1.0,
        controls=RobustMinimaConfig(),
    )

    assert metrics.recovered_psi_v2_per_m2 == 0.0
    assert metrics.distance_to_nearest_electrode_m > 1.0
    assert len(metrics.hessian_steps_m) == 4
    np.testing.assert_allclose(metrics.hessian_eigenvalues_v2_per_m4, 2.0)
    assert metrics.hessian_stable
    assert metrics.robust_accepted


def test_facet_sensitive_candidate_classification() -> None:
    assert classify_facet_sensitive(0.01, 0.75, False)
    assert not classify_facet_sensitive(0.01, 0.75, True)
    assert not classify_facet_sensitive(0.10, 0.75, False)


def test_hessian_stencil_stability_classification() -> None:
    stable = np.asarray([[2.0, 4.0], [2.2, 3.8], [1.9, 4.1], [2.1, 4.2]])
    classification, accepted, variation = classify_hessian_stability(stable)
    assert classification == "stable-positive-hessian"
    assert accepted
    assert variation < 2.0

    unstable = stable.copy()
    unstable[-1, 0] = -0.1
    classification, accepted, variation = classify_hessian_stability(unstable)
    assert classification == "unstable-hessian-signature"
    assert not accepted
    assert np.isinf(variation)


def test_robust_candidate_selection_and_rejected_preservation() -> None:
    recovered, potential = _synthetic_radial_field()
    base = compute_candidate_quality(
        candidate_id=1,
        seed=CandidateSeed(np.zeros(2), ("synthetic",), 1, True),
        recovered_field=recovered,
        potential_v=potential,
        candidate_psi_scale=1.0,
        controls=RobustMinimaConfig(),
    )
    candidates = (
        replace(base, candidate_id=1, position_m=np.asarray([0.10, 0.00]), recovered_psi_v2_per_m2=0.01),
        replace(base, candidate_id=2, position_m=np.asarray([-0.10, 0.00]), recovered_psi_v2_per_m2=0.02),
        replace(base, candidate_id=3, position_m=np.asarray([0.00, 0.10]), recovered_psi_v2_per_m2=0.03),
        replace(
            base,
            candidate_id=4,
            position_m=np.asarray([0.00, -0.10]),
            recovered_psi_v2_per_m2=0.001,
            robust_accepted=False,
            classification_reason="facet-sensitive-unstable-signature",
        ),
    )

    selected = select_robust_candidates(candidates, 3)
    result = MinimaModeResult("robust", selected, candidates, None, expected_minima=3)

    assert len(selected) == 3
    assert all(not np.array_equal(item.position_m, candidates[3].position_m) for item in selected)
    assert result.rejected_candidates == 1
    assert len(result.candidates) == 4
    assert result.candidates[3].classification_reason == "facet-sensitive-unstable-signature"


def test_explicit_raw_and_robust_modes_on_synthetic_field() -> None:
    recovered, potential = _synthetic_radial_field()
    search = MinimaSearchConfig(
        search_half_extent_m=0.8,
        coarse_grid_points_per_axis=21,
        optimizer_step_m=1.0e-4,
        merge_distance_m=0.05,
        hessian_step_m=0.02,
        expected_minima=1,
    )

    raw = run_minima_mode(
        recovered,
        potential,
        search,
        mode="raw-element-diagnostic",
    )
    robust = run_minima_mode(recovered, potential, search, mode="robust")

    assert raw.mode == "raw-element-diagnostic"
    assert raw.candidates
    assert robust.mode == "robust"
    assert robust.completed
    assert robust.candidates
    assert any("recovered-cell-zero" in item.source_names for item in robust.candidates)
