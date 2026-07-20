"""Mocked tests for Milestone 8 validation and branch summaries."""

from __future__ import annotations

import numpy as np

from rf_trap_forward.reference_validation import MinimumMatch
from rf_trap_forward.robust_validation import (
    ModeValidationRow,
    summarize_mode_rows,
    track_mesh_branches,
)


def _row(
    row_number: int,
    errors_m: tuple[float, float, float],
    *,
    mesh_size_m: float = 2.0e-3,
    positions_m: np.ndarray | None = None,
    topology_candidates: int = 3,
    selected_flags: int = 0,
    rejected: int = 0,
) -> ModeValidationRow:
    reference = np.asarray(((-1.0e-3, 0.0), (0.0, 1.0e-3), (1.0e-3, 0.0)))
    computed = reference.copy() if positions_m is None else np.asarray(positions_m, dtype=float)
    matches = tuple(
        MinimumMatch(
            reference_index=index,
            computed_index=index,
            reference_position_m=reference[index],
            computed_position_m=computed[index],
            delta_m=computed[index] - reference[index],
            distance_m=error,
        )
        for index, error in enumerate(errors_m)
    )
    return ModeValidationRow(
        mode_name="robust",
        mapping_name="identity",
        mesh_size_m=mesh_size_m,
        row_number=row_number,
        status="ok",
        reference_positions_m=reference,
        computed_positions_m=computed,
        matches=matches,
        topology_candidate_count=topology_candidates,
        selected_interpolation_sensitive=selected_flags,
        rejected_candidates=rejected,
        total_candidates=3 + rejected,
        node_count=100,
        triangle_count=180,
        relative_free_residual=1.0e-14,
        runtime_seconds=1.0,
    )


def test_old_vs_robust_summary_logic_and_gate() -> None:
    rows = (
        _row(1, (0.10e-3, 0.20e-3, 0.30e-3), rejected=2),
        _row(2, (0.10e-3, 0.15e-3, 0.20e-3), rejected=1),
    )
    summary = summarize_mode_rows(rows)

    assert summary.completed_rows == 2
    assert summary.exactly_three_rows == 2
    assert summary.matched_minima == 6
    assert summary.rejected_candidates == 3
    assert summary.validation_gate_passed

    failed = summarize_mode_rows(
        (rows[0], _row(2, (0.10e-3, 0.15e-3, 0.60e-3)))
    )
    assert not failed.validation_gate_passed


def test_mesh_branch_tracking_uses_spatial_assignment() -> None:
    coarse = _row(1, (0.0, 0.0, 0.0))
    permuted = np.asarray(((1.05e-3, 0.0), (-0.95e-3, 0.0), (0.0, 1.05e-3)))
    fine = _row(
        1,
        (0.0, 0.0, 0.0),
        mesh_size_m=1.0e-3,
        positions_m=permuted,
    )

    records = track_mesh_branches((coarse, fine), stability_tolerance_m=0.10e-3)

    assert len(records) == 3
    np.testing.assert_allclose(sorted(item.shift_m for item in records), 0.05e-3)
    assert all(item.stable for item in records)
