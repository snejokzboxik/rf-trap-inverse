"""Finite-element solution of the homogeneous Laplace equation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from skfem import Basis, ElementTriP1, asm, condense, solve
from skfem.models.poisson import laplace

from .config import SolverConfig
from .geometry import TrapGeometry
from .mesh import TrapMesh


@dataclass(frozen=True)
class FEMSolution:
    """Potential solution and numerical diagnostics on a P1 basis."""

    geometry: TrapGeometry
    trap_mesh: TrapMesh
    basis: Basis
    potential_v: NDArray[np.float64]
    relative_free_residual: float
    electrode_boundary_error_v: float
    outer_boundary_error_v: float


def solve_potential(
    geometry: TrapGeometry,
    trap_mesh: TrapMesh,
    config: SolverConfig,
) -> FEMSolution:
    """Solve Laplace's equation with electrode and outer Dirichlet values."""

    basis = Basis(trap_mesh.mesh, ElementTriP1())
    stiffness = csr_matrix(asm(laplace, basis))
    right_hand_side = np.zeros(basis.N, dtype=float)
    prescribed = np.zeros(basis.N, dtype=float)
    electrode_potentials = geometry.config.resolved_electrode_potentials_v
    for nodes, potential_v in zip(
        trap_mesh.electrode_boundary_nodes_by_electrode,
        electrode_potentials,
        strict=True,
    ):
        prescribed[nodes] = potential_v
    prescribed[trap_mesh.outer_boundary_nodes] = geometry.config.outer_potential_v
    dirichlet_nodes = np.union1d(
        trap_mesh.electrode_boundary_nodes,
        trap_mesh.outer_boundary_nodes,
    )
    potential = np.asarray(
        solve(
            *condense(
                stiffness,
                right_hand_side,
                x=prescribed,
                D=dirichlet_nodes,
            )
        ),
        dtype=float,
    )

    free_nodes = np.setdiff1d(np.arange(basis.N), dirichlet_nodes)
    residual = stiffness @ potential - right_hand_side
    forcing = stiffness[free_nodes][:, dirichlet_nodes] @ prescribed[dirichlet_nodes]
    denominator = max(float(np.linalg.norm(forcing)), np.finfo(float).tiny)
    relative_residual = float(np.linalg.norm(residual[free_nodes]) / denominator)
    electrode_error = max(
        float(np.max(np.abs(potential[nodes] - potential_v)))
        for nodes, potential_v in zip(
            trap_mesh.electrode_boundary_nodes_by_electrode,
            electrode_potentials,
            strict=True,
        )
    )
    outer_error = float(
        np.max(
            np.abs(
                potential[trap_mesh.outer_boundary_nodes]
                - geometry.config.outer_potential_v
            )
        )
    )
    _validate_solution(
        potential,
        geometry,
        config,
        relative_residual,
        electrode_error,
        outer_error,
    )
    return FEMSolution(
        geometry=geometry,
        trap_mesh=trap_mesh,
        basis=basis,
        potential_v=potential,
        relative_free_residual=relative_residual,
        electrode_boundary_error_v=electrode_error,
        outer_boundary_error_v=outer_error,
    )


def _validate_solution(
    potential_v: NDArray[np.float64],
    geometry: TrapGeometry,
    config: SolverConfig,
    relative_residual: float,
    electrode_error_v: float,
    outer_error_v: float,
) -> None:
    if not np.all(np.isfinite(potential_v)):
        raise RuntimeError("the FEM solution contains non-finite values")
    if relative_residual > config.relative_residual_tolerance:
        raise RuntimeError(
            f"relative free-node residual {relative_residual:.3e} exceeds tolerance"
        )
    if max(electrode_error_v, outer_error_v) > config.maximum_principle_tolerance_v:
        raise RuntimeError("Dirichlet boundary values were not imposed accurately")
    boundary_values = (
        *geometry.config.resolved_electrode_potentials_v,
        geometry.config.outer_potential_v,
    )
    lower = min(boundary_values)
    upper = max(boundary_values)
    tolerance = config.maximum_principle_tolerance_v
    if float(np.min(potential_v)) < lower - tolerance:
        raise RuntimeError("FEM potential violates the discrete minimum principle")
    if float(np.max(potential_v)) > upper + tolerance:
        raise RuntimeError("FEM potential violates the discrete maximum principle")
