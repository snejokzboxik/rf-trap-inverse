"""Tests for the named real-scale trap configuration."""

from __future__ import annotations

from math import sqrt

import numpy as np
import pytest

from rf_trap_forward.real_scale import (
    REAL_ELECTRODE_RADIUS_M,
    REAL_INNER_RADIUS_M,
    REAL_OUTER_BOUNDARY_RADIUS_M,
    electrode_center_radius_m,
    real_scale_forward_config,
    real_scale_geometry_config,
)


def test_inner_radius_converts_to_electrode_center_radius() -> None:
    """The supplied surface clearance plus electrode radius must equal 21.48 mm."""

    assert electrode_center_radius_m(
        REAL_INNER_RADIUS_M,
        REAL_ELECTRODE_RADIUS_M,
    ) == pytest.approx(21.48e-3)


def test_real_scale_electrode_centers_follow_documented_numbering() -> None:
    """E1--E4 must occupy upper-left, upper-right, lower-left, lower-right."""

    geometry = real_scale_geometry_config()
    a = 21.48e-3 / sqrt(2.0)
    np.testing.assert_allclose(
        geometry.nominal_centers_m,
        ((-a, +a), (+a, +a), (-a, -a), (+a, -a)),
    )
    radii = np.linalg.norm(geometry.nominal_centers_m, axis=1)
    np.testing.assert_allclose(radii - geometry.electrode_radius_m, REAL_INNER_RADIUS_M)
    assert geometry.outer_radius_m == REAL_OUTER_BOUNDARY_RADIUS_M


def test_real_scale_search_contains_supplied_reference_radius() -> None:
    """The default square must extend beyond the dataset's 6.5 mm radial scale."""

    config = real_scale_forward_config()
    assert config.minima.search_half_extent_m >= 6.5e-3
    assert config.minima.search_half_extent_m == pytest.approx(8.0e-3)
