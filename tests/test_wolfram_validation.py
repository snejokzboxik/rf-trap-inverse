"""Focused tests for the Wolfram displacement convention."""

from __future__ import annotations

import numpy as np

from rf_trap_forward.absolute_validation import (
    WOLFRAM_ELECTRODE_MAPPING,
    wolfram_to_fem_absolute_displacements_m,
)
from rf_trap_forward.config import GeometryConfig
from rf_trap_forward.geometry import (
    absolute_displacements_m,
    build_geometry_from_absolute_displacements,
)


def _raw_wolfram_pairs() -> np.ndarray:
    return np.asarray(((1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)))


def _geometry_config() -> GeometryConfig:
    return GeometryConfig(
        electrode_radius_m=0.1,
        nominal_centers_m=((-0.5, 0.5), (0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)),
        outer_radius_m=3.0,
    )


def test_wolfram_reorder_is_w3_w1_w4_w2() -> None:
    raw = _raw_wolfram_pairs()
    transformed = wolfram_to_fem_absolute_displacements_m(raw)
    source_indices = np.asarray(WOLFRAM_ELECTRODE_MAPPING) - 1
    np.testing.assert_allclose(-transformed, raw[source_indices])


def test_wolfram_displacements_receive_global_sign_flip() -> None:
    transformed = wolfram_to_fem_absolute_displacements_m(_raw_wolfram_pairs())
    assert np.all(transformed < 0.0)


def test_wolfram_transform_moves_all_four_fem_electrodes() -> None:
    config = _geometry_config()
    transformed = 1.0e-3 * wolfram_to_fem_absolute_displacements_m(
        _raw_wolfram_pairs()
    )
    geometry = build_geometry_from_absolute_displacements(config, transformed)
    np.testing.assert_allclose(
        geometry.centers_m,
        np.asarray(config.nominal_centers_m) + transformed,
    )
    assert np.all(np.linalg.norm(transformed, axis=1) > 0.0)


def test_raw_absolute_path_remains_available() -> None:
    raw = 1.0e-3 * _raw_wolfram_pairs()
    np.testing.assert_allclose(absolute_displacements_m(raw), raw)
    geometry = build_geometry_from_absolute_displacements(_geometry_config(), raw)
    np.testing.assert_allclose(
        geometry.centers_m,
        np.asarray(_geometry_config().nominal_centers_m) + raw,
    )
