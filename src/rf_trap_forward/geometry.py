"""Geometry construction and analytic domain-membership checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import GeometryConfig, displacement_vector_m


@dataclass(frozen=True)
class TrapGeometry:
    """A validated four-electrode geometry for one displacement vector."""

    config: GeometryConfig
    centers_m: NDArray[np.float64]
    displacements_m: NDArray[np.float64]

    def contains_points(self, points_m: ArrayLike, clearance_m: float = 0.0) -> NDArray[np.bool_]:
        """Return whether points lie in vacuum, optionally away from boundaries.

        Positive clearance shrinks the admissible outer disk and expands each
        excluded electrode disk by the same SI distance.
        """

        points = np.asarray(points_m, dtype=float)
        if points.ndim == 1:
            points = points[np.newaxis, :]
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points_m must have shape (2,) or (n, 2)")
        if clearance_m < 0.0 or not np.isfinite(clearance_m):
            raise ValueError("clearance_m must be finite and non-negative")
        outer_distance = np.linalg.norm(points, axis=1)
        inside_outer = outer_distance < self.config.outer_radius_m - clearance_m
        electrode_distances = np.linalg.norm(
            points[:, np.newaxis, :] - self.centers_m[np.newaxis, :, :],
            axis=2,
        )
        outside_electrodes = np.all(
            electrode_distances > self.config.electrode_radius_m + clearance_m,
            axis=1,
        )
        return inside_outer & outside_electrodes


@dataclass(frozen=True)
class GeometrySanity:
    """Clearance diagnostics for one four-electrode geometry."""

    minimum_electrode_gap_m: float
    minimum_outer_clearance_m: float
    pairwise_center_distances_m: NDArray[np.float64]
    valid: bool


def geometry_sanity(
    config: GeometryConfig,
    centers_m: ArrayLike,
) -> GeometrySanity:
    """Return pair and outer-boundary clearances without hiding invalid cases."""

    centers = np.asarray(centers_m, dtype=float)
    if centers.shape != (4, 2) or not np.all(np.isfinite(centers)):
        raise ValueError("centers_m must contain four finite (x, y) pairs")
    radius = config.electrode_radius_m
    separation = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    distances = np.linalg.norm(separation, axis=2)[np.triu_indices(4, k=1)]
    electrode_gap = float(np.min(distances - 2.0 * radius))
    outer_clearance = float(
        np.min(config.outer_radius_m - (np.linalg.norm(centers, axis=1) + radius))
    )
    return GeometrySanity(
        minimum_electrode_gap_m=electrode_gap,
        minimum_outer_clearance_m=outer_clearance,
        pairwise_center_distances_m=distances,
        valid=electrode_gap > 0.0 and outer_clearance > 0.0,
    )


def build_geometry(
    config: GeometryConfig,
    displacements_m: NDArray[np.floating] | list[float] | tuple[float, ...],
) -> TrapGeometry:
    """Apply displacements to electrodes 2--4 and validate the resulting domain."""

    vector = displacement_vector_m(displacements_m)
    centers = np.asarray(config.nominal_centers_m, dtype=float).copy()
    centers[1:] += vector.reshape(3, 2)
    _validate_non_overlapping_domain(config, centers)
    return TrapGeometry(config=config, centers_m=centers, displacements_m=vector)


def absolute_displacements_m(values: ArrayLike) -> NDArray[np.float64]:
    """Return four validated absolute electrode-displacement pairs in metres.

    Both an explicit ``(4, 2)`` array and a flat eight-component vector are
    accepted. The legacy six-component, E1-fixed representation is
    intentionally rejected by this Data.txt input path.
    """

    displacements = np.asarray(values, dtype=float)
    if displacements.shape == (8,):
        displacements = displacements.reshape(4, 2)
    if displacements.shape != (4, 2):
        raise ValueError(
            "absolute_displacements_m must have shape (4, 2) or (8,)"
        )
    if not np.all(np.isfinite(displacements)):
        raise ValueError("absolute_displacements_m must contain only finite values")
    return displacements.copy()


def build_geometry_from_absolute_displacements(
    config: GeometryConfig,
    absolute_displacements_m_values: ArrayLike,
) -> TrapGeometry:
    """Move all four electrodes in the fixed outer-boundary coordinate frame."""

    displacements = absolute_displacements_m(absolute_displacements_m_values)
    centers = np.asarray(config.nominal_centers_m, dtype=float) + displacements
    _validate_non_overlapping_domain(config, centers)
    return TrapGeometry(
        config=config,
        centers_m=centers,
        displacements_m=displacements,
    )


def _validate_non_overlapping_domain(config: GeometryConfig, centers_m: NDArray[np.float64]) -> None:
    sanity = geometry_sanity(config, centers_m)
    if sanity.minimum_outer_clearance_m <= 0.0:
        raise ValueError("every electrode must lie strictly inside the outer boundary")
    if sanity.minimum_electrode_gap_m <= 0.0:
        raise ValueError("electrode disks must not touch or overlap")
