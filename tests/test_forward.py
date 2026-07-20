"""End-to-end acceptance test for the one-configuration milestone."""

from __future__ import annotations

import numpy as np


def test_forward_pipeline_returns_three_valid_angle_sorted_minima(forward_result) -> None:
    """The demonstrator must produce the required validated three-point output."""

    result = forward_result
    assert len(result.minima) == 3
    positions = result.minima_positions_m()
    assert positions.shape == (3, 2)
    assert np.all(result.geometry.contains_points(positions))
    angles = np.asarray([minimum.polar_angle_rad for minimum in result.minima])
    assert np.all(np.diff(angles) >= 0.0)
    for minimum in result.minima:
        assert minimum.pseudopotential_v2_per_m2 >= 0.0
        assert np.all(minimum.hessian_eigenvalues_v2_per_m4 > 0.0)
    assert result.minima_diagnostics.hessian_validated_candidates >= 3
    assert (
        len(result.minima_diagnostics.hessian_validated_minima)
        == result.minima_diagnostics.hessian_validated_candidates
    )
