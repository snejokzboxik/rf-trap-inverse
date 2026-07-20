"""Explicitly provisional geometry used by examples and validation commands."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import (
    ForwardModelConfig,
    GeometryConfig,
    MeshConfig,
    MinimaSearchConfig,
    SolverConfig,
)


def demonstrator_config() -> ForwardModelConfig:
    """Return the provisional milestone configuration in SI units."""

    return ForwardModelConfig(
        geometry=GeometryConfig(
            electrode_radius_m=0.32e-3,
            nominal_centers_m=(
                (1.10e-3, 0.0),
                (0.0, 1.10e-3),
                (-1.10e-3, 0.0),
                (0.0, -1.10e-3),
            ),
            outer_radius_m=4.0e-3,
        ),
        mesh=MeshConfig(characteristic_length_m=0.08e-3),
        solver=SolverConfig(),
        minima=MinimaSearchConfig(
            search_half_extent_m=0.70e-3,
            coarse_grid_points_per_axis=71,
            merge_distance_m=0.02e-3,
            hessian_step_m=0.004e-3,
        ),
    )


def demonstrator_displacements_m() -> NDArray[np.float64]:
    """Return the single asymmetric milestone displacement vector in metres."""

    return np.asarray(
        [120.0, -80.0, -150.0, 110.0, 90.0, 160.0],
        dtype=float,
    ) * 1.0e-6
