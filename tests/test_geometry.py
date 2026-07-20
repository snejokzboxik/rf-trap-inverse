"""Unit tests for displacement semantics and domain validation."""

from __future__ import annotations

import numpy as np
import pytest

from rf_trap_forward import GeometryConfig
from rf_trap_forward.geometry import build_geometry


def test_electrode_one_is_fixed_and_other_displacements_are_applied() -> None:
    """Electrode 1 must define the fixed reference for all six inputs."""

    config = GeometryConfig(
        electrode_radius_m=0.1,
        nominal_centers_m=((0.5, 0.0), (0.0, 0.5), (-0.5, 0.0), (0.0, -0.5)),
        outer_radius_m=2.0,
    )
    displacement = np.asarray([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    geometry = build_geometry(config, displacement)
    expected = np.asarray(
        ((0.5, 0.0), (0.01, 0.52), (-0.47, 0.04), (0.05, -0.44))
    )
    np.testing.assert_allclose(geometry.centers_m, expected)


def test_overlapping_displaced_electrodes_are_rejected() -> None:
    """A displacement that causes touching conductors must fail explicitly."""

    config = GeometryConfig(
        electrode_radius_m=0.2,
        nominal_centers_m=((0.5, 0.0), (0.0, 0.5), (-0.5, 0.0), (0.0, -0.5)),
        outer_radius_m=2.0,
    )
    with pytest.raises(ValueError, match="must not touch or overlap"):
        build_geometry(config, [0.5, -0.5, 0.0, 0.0, 0.0, 0.0])


def test_domain_membership_excludes_electrode_interiors(geometry) -> None:
    """Analytic membership must reject conductor centres and accept the origin."""

    assert bool(geometry.contains_points([0.0, 0.0])[0])
    assert not np.any(geometry.contains_points(geometry.centers_m))

