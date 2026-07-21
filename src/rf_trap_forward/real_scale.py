"""Named real-scale trap geometry and coarse validation configurations."""

from __future__ import annotations

from math import sqrt

import numpy as np

from .config import (
    ForwardModelConfig,
    GeometryConfig,
    MeshConfig,
    MeshSizeFieldConfig,
    MinimaSearchConfig,
    SolverConfig,
)

REAL_OUTER_BOUNDARY_RADIUS_M = 50.0e-3
REAL_ELECTRODE_RADIUS_M = 10.0e-3
REAL_INNER_RADIUS_M = 11.48e-3
REAL_SEARCH_HALF_WIDTH_M = 8.0e-3
REAL_COARSE_MESH_SIZES_M = (2.0e-3, 1.5e-3, 1.0e-3)
REAL_CENTRAL_REFINEMENT_RADIUS_M = 8.0e-3
REAL_ELECTRODE_BOUNDARY_MESH_SIZE_M = 0.50e-3
DIAGONAL_ALTERNATING_POTENTIALS_V = (1.0, -1.0, -1.0, 1.0)


def electrode_center_radius_m(inner_radius_m: float, electrode_radius_m: float) -> float:
    """Convert center-to-surface clearance and electrode radius to center radius."""

    values = np.asarray([inner_radius_m, electrode_radius_m], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("inner and electrode radii must be finite and positive")
    return float(inner_radius_m + electrode_radius_m)


def real_scale_geometry_config(
    *,
    electrode_potentials_v: tuple[float, float, float, float] | None = None,
) -> GeometryConfig:
    """Return the supplied 50 mm real-scale diagonal electrode geometry.

    Numbering is E1 upper-left, E2 upper-right, E3 lower-left, and E4
    lower-right. ``electrode_potentials_v=None`` retains the core all-``+1 V``
    model. The diagnostic alternating checkerboard is ``(+1, -1, -1, +1)``.
    """

    center_radius = electrode_center_radius_m(
        REAL_INNER_RADIUS_M,
        REAL_ELECTRODE_RADIUS_M,
    )
    diagonal_coordinate = center_radius / sqrt(2.0)
    return GeometryConfig(
        electrode_radius_m=REAL_ELECTRODE_RADIUS_M,
        nominal_centers_m=(
            (-diagonal_coordinate, +diagonal_coordinate),
            (+diagonal_coordinate, +diagonal_coordinate),
            (-diagonal_coordinate, -diagonal_coordinate),
            (+diagonal_coordinate, -diagonal_coordinate),
        ),
        outer_radius_m=REAL_OUTER_BOUNDARY_RADIUS_M,
        electrode_potentials_v=electrode_potentials_v,
    )


def real_scale_forward_config(
    *,
    mesh_size_m: float = REAL_COARSE_MESH_SIZES_M[0],
    search_half_width_m: float = REAL_SEARCH_HALF_WIDTH_M,
    electrode_potentials_v: tuple[float, float, float, float] | None = None,
) -> ForwardModelConfig:
    """Return a coarse first-pass FEM configuration for the real trap scale."""

    if not np.isfinite(search_half_width_m) or search_half_width_m < 6.5e-3:
        raise ValueError("search_half_width_m must be finite and at least 6.5 mm")
    return ForwardModelConfig(
        geometry=real_scale_geometry_config(
            electrode_potentials_v=electrode_potentials_v
        ),
        mesh=MeshConfig(characteristic_length_m=mesh_size_m),
        solver=SolverConfig(),
        minima=MinimaSearchConfig(
            search_half_extent_m=search_half_width_m,
            coarse_grid_points_per_axis=81,
            optimizer_step_m=5.0e-6,
            merge_distance_m=0.10e-3,
            hessian_step_m=0.05e-3,
        ),
    )


def locally_refined_real_scale_forward_config(
    *,
    central_mesh_size_m: float,
    outer_mesh_size_m: float = REAL_COARSE_MESH_SIZES_M[0],
    electrode_boundary_mesh_size_m: float = REAL_ELECTRODE_BOUNDARY_MESH_SIZE_M,
    central_region_radius_m: float = REAL_CENTRAL_REFINEMENT_RADIUS_M,
    search_half_width_m: float = REAL_SEARCH_HALF_WIDTH_M,
    electrode_potentials_v: tuple[float, float, float, float] | None = None,
    geometry: GeometryConfig | None = None,
) -> ForwardModelConfig:
    """Return the named Milestone 9 central-refinement configuration.

    The supplied real-scale geometry remains the default. Passing ``geometry``
    makes geometry calibration explicit and never mutates the legacy/default
    :func:`real_scale_forward_config` configuration.
    """

    if not np.isfinite(search_half_width_m) or search_half_width_m < 6.5e-3:
        raise ValueError("search_half_width_m must be finite and at least 6.5 mm")
    size_field = MeshSizeFieldConfig(
        outer_mesh_size_m=outer_mesh_size_m,
        electrode_boundary_mesh_size_m=electrode_boundary_mesh_size_m,
        central_region_radius_m=central_region_radius_m,
        central_mesh_size_m=central_mesh_size_m,
        central_transition_width_m=1.0e-3,
        electrode_transition_width_m=2.0e-3,
    )
    selected_geometry = geometry or real_scale_geometry_config(
        electrode_potentials_v=electrode_potentials_v
    )
    if geometry is not None and electrode_potentials_v is not None:
        raise ValueError("pass either geometry or electrode_potentials_v, not both")
    return ForwardModelConfig(
        geometry=selected_geometry,
        mesh=MeshConfig(
            characteristic_length_m=outer_mesh_size_m,
            size_field=size_field,
        ),
        solver=SolverConfig(),
        minima=MinimaSearchConfig(
            search_half_extent_m=search_half_width_m,
            coarse_grid_points_per_axis=81,
            optimizer_step_m=max(2.0e-7, min(5.0e-6, central_mesh_size_m / 10.0)),
            merge_distance_m=max(2.0e-6, min(0.10e-3, central_mesh_size_m / 2.0)),
            hessian_step_m=max(1.0e-6, min(0.05e-3, central_mesh_size_m / 4.0)),
        ),
    )
