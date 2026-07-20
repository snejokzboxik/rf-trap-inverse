"""Finite-element solution of the homogeneous Laplace equation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix
from skfem import Basis, ElementTriP1, MeshTri, asm, condense, solve
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


@dataclass(frozen=True)
class DirichletLaplaceSolution:
    """Generic P1 Laplace solution used by production and analytic audits."""

    basis: Basis
    potential_v: NDArray[np.float64]
    relative_free_residual: float
    boundary_error_v: float


def solve_dirichlet_laplace(
    mesh: MeshTri,
    dirichlet_nodes: ArrayLike,
    prescribed_values_v: ArrayLike,
    config: SolverConfig,
) -> DirichletLaplaceSolution:
    """Solve homogeneous Laplace data on a P1 mesh with nodal Dirichlet values.

    ``prescribed_values_v`` contains one value per mesh vertex; only entries at
    ``dirichlet_nodes`` are imposed. This is the shared linear-system path used
    by the trap solver and the analytic numerical-audit problems.
    """

    nodes = np.asarray(dirichlet_nodes, dtype=np.int64)
    prescribed = np.asarray(prescribed_values_v, dtype=float)
    number_of_nodes = mesh.p.shape[1]
    if nodes.ndim != 1 or nodes.size == 0 or np.unique(nodes).size != nodes.size:
        raise ValueError("dirichlet_nodes must be a nonempty unique vector")
    if np.any(nodes < 0) or np.any(nodes >= number_of_nodes):
        raise ValueError("dirichlet_nodes contains an out-of-range vertex")
    if prescribed.shape != (number_of_nodes,) or not np.all(np.isfinite(prescribed)):
        raise ValueError("prescribed_values_v must contain one finite value per vertex")

    basis = Basis(mesh, ElementTriP1())
    stiffness = csr_matrix(asm(laplace, basis))
    right_hand_side = np.zeros(basis.N, dtype=float)
    potential = np.asarray(
        solve(
            *condense(
                stiffness,
                right_hand_side,
                x=prescribed,
                D=nodes,
            )
        ),
        dtype=float,
    )
    free_nodes = np.setdiff1d(np.arange(basis.N), nodes)
    residual = stiffness @ potential - right_hand_side
    forcing = stiffness[free_nodes][:, nodes] @ prescribed[nodes]
    denominator = max(float(np.linalg.norm(forcing)), np.finfo(float).tiny)
    relative_residual = float(np.linalg.norm(residual[free_nodes]) / denominator)
    boundary_error = float(np.max(np.abs(potential[nodes] - prescribed[nodes])))
    _validate_generic_solution(
        potential,
        prescribed[nodes],
        config,
        relative_residual,
        boundary_error,
    )
    return DirichletLaplaceSolution(
        basis=basis,
        potential_v=potential,
        relative_free_residual=relative_residual,
        boundary_error_v=boundary_error,
    )


def solve_potential(
    geometry: TrapGeometry,
    trap_mesh: TrapMesh,
    config: SolverConfig,
) -> FEMSolution:
    """Solve Laplace's equation with electrode and outer Dirichlet values."""

    prescribed = np.zeros(trap_mesh.mesh.p.shape[1], dtype=float)
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
    generic = solve_dirichlet_laplace(
        trap_mesh.mesh,
        dirichlet_nodes,
        prescribed,
        config,
    )
    potential = generic.potential_v
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
    return FEMSolution(
        geometry=geometry,
        trap_mesh=trap_mesh,
        basis=generic.basis,
        potential_v=potential,
        relative_free_residual=generic.relative_free_residual,
        electrode_boundary_error_v=electrode_error,
        outer_boundary_error_v=outer_error,
    )


def _validate_generic_solution(
    potential_v: NDArray[np.float64],
    boundary_values_v: NDArray[np.float64],
    config: SolverConfig,
    relative_residual: float,
    boundary_error_v: float,
) -> None:
    if not np.all(np.isfinite(potential_v)):
        raise RuntimeError("the FEM solution contains non-finite values")
    if relative_residual > config.relative_residual_tolerance:
        raise RuntimeError(
            f"relative free-node residual {relative_residual:.3e} exceeds tolerance"
        )
    if boundary_error_v > config.maximum_principle_tolerance_v:
        raise RuntimeError("Dirichlet boundary values were not imposed accurately")
    lower = float(np.min(boundary_values_v))
    upper = float(np.max(boundary_values_v))
    tolerance = config.maximum_principle_tolerance_v
    if float(np.min(potential_v)) < lower - tolerance:
        raise RuntimeError("FEM potential violates the discrete minimum principle")
    if float(np.max(potential_v)) > upper + tolerance:
        raise RuntimeError("FEM potential violates the discrete maximum principle")
