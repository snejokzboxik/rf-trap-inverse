"""Electric-field recovery and pseudopotential evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from skfem import Basis, ElementTriP1, MeshTri

from .geometry import TrapGeometry
from .solver import FEMSolution


class PointOutsideVacuumError(ValueError):
    """Raised when field evaluation is requested outside the vacuum domain."""


@dataclass
class RecoveredField:
    """Continuous P1 surrogate of the electric field on the FEM mesh.

    The exact gradient of a P1 potential is constant per triangle and jumps at
    facets.  This surrogate area-averages adjacent element gradients at vertices
    and linearly interpolates the recovered nodal values.  It is intended for
    locating candidate field zeros, not for replacing mesh-convergence studies.
    """

    geometry: TrapGeometry
    mesh: MeshTri
    electric_field_nodes_v_per_m: NDArray[np.float64]
    _interpolators: tuple[Callable[[NDArray[np.float64]], NDArray[np.float64]], ...] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Construct reusable component interpolators after validation."""

        values = np.asarray(self.electric_field_nodes_v_per_m, dtype=float)
        if values.shape != (2, self.mesh.p.shape[1]):
            raise ValueError("electric_field_nodes_v_per_m must have shape (2, n_nodes)")
        if not np.all(np.isfinite(values)):
            raise ValueError("electric_field_nodes_v_per_m must be finite")
        self.electric_field_nodes_v_per_m = values
        basis = Basis(self.mesh, ElementTriP1())
        self._interpolators = (
            basis.interpolator(values[0]),
            basis.interpolator(values[1]),
        )

    def evaluate(self, points_m: ArrayLike) -> NDArray[np.float64]:
        """Evaluate recovered electric-field vectors at valid vacuum points."""

        points = np.asarray(points_m, dtype=float)
        single_point = points.ndim == 1
        if single_point:
            points = points[np.newaxis, :]
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points_m must have shape (2,) or (n, 2)")
        if not np.all(self.geometry.contains_points(points)):
            raise PointOutsideVacuumError("field points must lie in the vacuum domain")
        try:
            components = np.column_stack(
                tuple(interpolator(points.T) for interpolator in self._interpolators)
            )
        except ValueError as error:
            raise PointOutsideVacuumError(
                "a point lies outside the polygonal FEM approximation"
            ) from error
        return components[0] if single_point else components

    def pseudopotential(self, points_m: ArrayLike) -> NDArray[np.float64] | float:
        """Evaluate the normalized pseudopotential ``|E|^2`` in V^2/m^2."""

        field_values = self.evaluate(points_m)
        if field_values.ndim == 1:
            return float(np.dot(field_values, field_values))
        return np.einsum("ij,ij->i", field_values, field_values)


def recover_nodal_electric_field(
    mesh: MeshTri,
    potential_v: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Recover ``-grad(phi)`` at vertices by triangle-area-weighted averaging."""

    element_fields = element_electric_fields(mesh, potential_v)
    triangles = mesh.t
    points = mesh.p
    x0, y0 = points[:, triangles[0]]
    x1, y1 = points[:, triangles[1]]
    x2, y2 = points[:, triangles[2]]
    determinant = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    areas = 0.5 * np.abs(determinant)
    accumulated = np.zeros((2, points.shape[1]), dtype=float)
    weights = np.zeros(points.shape[1], dtype=float)
    for local_vertex in range(3):
        nodes = triangles[local_vertex]
        np.add.at(accumulated[0], nodes, areas * element_fields[0])
        np.add.at(accumulated[1], nodes, areas * element_fields[1])
        np.add.at(weights, nodes, areas)
    if np.any(weights <= 0.0):
        raise ValueError("mesh contains a vertex with no adjacent triangle area")
    return accumulated / weights


def element_electric_fields(
    mesh: MeshTri,
    potential_v: ArrayLike,
) -> NDArray[np.float64]:
    """Return exact per-triangle ``-grad(phi)`` for a nodal P1 potential."""

    potential = np.asarray(potential_v, dtype=float)
    if potential.shape != (mesh.p.shape[1],):
        raise ValueError("potential_v must contain one value per mesh vertex")
    triangles = mesh.t
    points = mesh.p
    x0, y0 = points[:, triangles[0]]
    x1, y1 = points[:, triangles[1]]
    x2, y2 = points[:, triangles[2]]
    determinant = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    if np.any(np.isclose(determinant, 0.0, rtol=0.0, atol=np.finfo(float).tiny)):
        raise ValueError("mesh contains a degenerate triangle")

    grad_shape_0 = np.vstack((y1 - y2, x2 - x1)) / determinant
    grad_shape_1 = np.vstack((y2 - y0, x0 - x2)) / determinant
    grad_shape_2 = np.vstack((y0 - y1, x1 - x0)) / determinant
    element_gradients = (
        potential[triangles[0]] * grad_shape_0
        + potential[triangles[1]] * grad_shape_1
        + potential[triangles[2]] * grad_shape_2
    )
    return -element_gradients


def recover_field(solution: FEMSolution) -> RecoveredField:
    """Build the documented continuous field surrogate from an FEM solution."""

    nodal_field = recover_nodal_electric_field(
        solution.trap_mesh.mesh,
        solution.potential_v,
    )
    return RecoveredField(
        geometry=solution.geometry,
        mesh=solution.trap_mesh.mesh,
        electric_field_nodes_v_per_m=nodal_field,
    )
