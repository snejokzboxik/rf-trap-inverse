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


def _validate_non_overlapping_domain(config: GeometryConfig, centers_m: NDArray[np.float64]) -> None:
    radius = config.electrode_radius_m
    radial_extent = np.linalg.norm(centers_m, axis=1) + radius
    if np.any(radial_extent >= config.outer_radius_m):
        raise ValueError("every electrode must lie strictly inside the outer boundary")
    separation = centers_m[:, np.newaxis, :] - centers_m[np.newaxis, :, :]
    distances = np.linalg.norm(separation, axis=2)
    upper_triangle = distances[np.triu_indices(4, k=1)]
    if np.any(upper_triangle <= 2.0 * radius):
        raise ValueError("electrode disks must not touch or overlap")

