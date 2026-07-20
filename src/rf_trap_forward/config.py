"""Configuration objects for the RF-trap forward model.

All lengths use metres and all electrical potentials use volts.  Unit-bearing
suffixes are deliberately included in field names to make accidental use of
micrometres difficult.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class GeometryConfig:
    """Physical geometry and Dirichlet data in SI units.

    The outer boundary is a circle centred at the origin.  The four nominal
    centres are ordered by electrode number; electrode 1 defines the reference
    frame and is therefore never displaced by the six-component model input.
    """

    electrode_radius_m: float
    nominal_centers_m: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]
    outer_radius_m: float
    electrode_potential_v: float = 1.0
    outer_potential_v: float = 0.0
    electrode_potentials_v: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        """Validate the immutable configuration after construction."""

        if not np.isfinite(self.electrode_radius_m) or self.electrode_radius_m <= 0.0:
            raise ValueError("electrode_radius_m must be finite and positive")
        if not np.isfinite(self.outer_radius_m) or self.outer_radius_m <= 0.0:
            raise ValueError("outer_radius_m must be finite and positive")
        centers = np.asarray(self.nominal_centers_m, dtype=float)
        if centers.shape != (4, 2) or not np.all(np.isfinite(centers)):
            raise ValueError("nominal_centers_m must contain four finite (x, y) pairs")
        if not np.isfinite(self.electrode_potential_v):
            raise ValueError("electrode_potential_v must be finite")
        if not np.isfinite(self.outer_potential_v):
            raise ValueError("outer_potential_v must be finite")
        if self.electrode_potentials_v is not None:
            potentials = np.asarray(self.electrode_potentials_v, dtype=float)
            if potentials.shape != (4,) or not np.all(np.isfinite(potentials)):
                raise ValueError(
                    "electrode_potentials_v must contain four finite values"
                )

    @property
    def resolved_electrode_potentials_v(self) -> tuple[float, float, float, float]:
        """Return one Dirichlet value per electrode in numbering order."""

        if self.electrode_potentials_v is not None:
            return self.electrode_potentials_v
        return (self.electrode_potential_v,) * 4


@dataclass(frozen=True)
class MeshConfig:
    """Deterministic first-order triangular mesh settings in SI units."""

    characteristic_length_m: float
    boundary_tolerance_m: float = 1.0e-9
    gmsh_algorithm: int = 6
    random_seed: int = 1
    random_factor: float = 0.0
    reproducible: bool = True

    def __post_init__(self) -> None:
        """Validate the mesh controls after construction."""

        if not np.isfinite(self.characteristic_length_m) or self.characteristic_length_m <= 0.0:
            raise ValueError("characteristic_length_m must be finite and positive")
        if not np.isfinite(self.boundary_tolerance_m) or self.boundary_tolerance_m <= 0.0:
            raise ValueError("boundary_tolerance_m must be finite and positive")
        if self.gmsh_algorithm <= 0:
            raise ValueError("gmsh_algorithm must be positive")
        if self.random_seed <= 0:
            raise ValueError("random_seed must be positive for reproducible Gmsh runs")
        if not np.isfinite(self.random_factor) or self.random_factor < 0.0:
            raise ValueError("random_factor must be finite and non-negative")


@dataclass(frozen=True)
class SolverConfig:
    """Acceptance limits for the finite-element linear solve."""

    relative_residual_tolerance: float = 1.0e-10
    maximum_principle_tolerance_v: float = 1.0e-10

    def __post_init__(self) -> None:
        """Validate solver tolerances after construction."""

        if self.relative_residual_tolerance <= 0.0:
            raise ValueError("relative_residual_tolerance must be positive")
        if self.maximum_principle_tolerance_v < 0.0:
            raise ValueError("maximum_principle_tolerance_v must be non-negative")


@dataclass(frozen=True)
class MinimaSearchConfig:
    """Controls for coarse detection, refinement, merging, and validation."""

    search_half_extent_m: float
    coarse_grid_points_per_axis: int = 61
    candidate_neighborhood: int = 3
    maximum_candidates: int = 24
    optimizer_step_m: float = 2.0e-7
    optimizer_ftol: float = 1.0e-12
    optimizer_max_iterations: int = 300
    optimizer_max_line_search_steps: int = 50
    merge_distance_m: float = 2.0e-5
    hessian_step_m: float = 2.0e-6
    minimum_hessian_eigenvalue_v2_per_m4: float = 0.0
    invalid_objective_v2_per_m2: float = 1.0e30
    expected_minima: int = 3

    def __post_init__(self) -> None:
        """Validate all search and finite-difference controls."""

        positive_lengths = (
            self.search_half_extent_m,
            self.optimizer_step_m,
            self.merge_distance_m,
            self.hessian_step_m,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive_lengths):
            raise ValueError("all minima-search lengths must be finite and positive")
        if self.coarse_grid_points_per_axis < 5:
            raise ValueError("coarse_grid_points_per_axis must be at least 5")
        if self.candidate_neighborhood < 3 or self.candidate_neighborhood % 2 == 0:
            raise ValueError("candidate_neighborhood must be an odd integer of at least 3")
        if self.maximum_candidates < self.expected_minima:
            raise ValueError("maximum_candidates must be at least expected_minima")
        if (
            self.optimizer_ftol <= 0.0
            or self.optimizer_max_iterations <= 0
            or self.optimizer_max_line_search_steps <= 0
        ):
            raise ValueError("optimizer tolerances and iteration limit must be positive")
        if self.minimum_hessian_eigenvalue_v2_per_m4 < 0.0:
            raise ValueError("minimum Hessian eigenvalue must be non-negative")
        if not np.isfinite(self.invalid_objective_v2_per_m2):
            raise ValueError("invalid_objective_v2_per_m2 must be finite")
        if self.invalid_objective_v2_per_m2 <= 0.0:
            raise ValueError("invalid_objective_v2_per_m2 must be positive")
        if self.expected_minima <= 0:
            raise ValueError("expected_minima must be positive")


@dataclass(frozen=True)
class ForwardModelConfig:
    """Complete configuration for one forward-model evaluation."""

    geometry: GeometryConfig
    mesh: MeshConfig
    solver: SolverConfig
    minima: MinimaSearchConfig


def displacement_vector_m(values: NDArray[np.floating] | list[float] | tuple[float, ...]) -> NDArray[np.float64]:
    """Return a validated six-component displacement vector in metres."""

    vector = np.asarray(values, dtype=float)
    if vector.shape != (6,):
        raise ValueError("displacements_m must have shape (6,)")
    if not np.all(np.isfinite(vector)):
        raise ValueError("displacements_m must contain only finite values")
    return vector
