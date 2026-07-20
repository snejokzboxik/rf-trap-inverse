"""Run the milestone-one demonstrator for one asymmetric displacement vector."""

from __future__ import annotations

import numpy as np

from rf_trap_forward import (
    ForwardModelConfig,
    GeometryConfig,
    MeshConfig,
    MinimaSearchConfig,
    SolverConfig,
    run_forward_model,
)


def example_config() -> ForwardModelConfig:
    """Return an explicitly provisional, SI-unit demonstrator configuration."""

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


def main() -> None:
    """Execute the forward model and print its physical and numerical outputs."""

    displacements_m = np.asarray(
        [120.0, -80.0, -150.0, 110.0, 90.0, 160.0],
        dtype=float,
    ) * 1.0e-6
    result = run_forward_model(displacements_m, example_config())

    print(f"mesh nodes: {result.trap_mesh.number_of_nodes}")
    print(f"mesh triangles: {result.trap_mesh.number_of_triangles}")
    print(f"relative free residual: {result.fem_solution.relative_free_residual:.3e}")
    print(f"search diagnostics: {result.minima_diagnostics}")
    for index, minimum in enumerate(result.minima, start=1):
        print(
            f"minimum {index}: position_m={minimum.position_m}, "
            f"position_um={minimum.position_m * 1.0e6}, "
            f"|E|^2={minimum.pseudopotential_v2_per_m2:.6e}, "
            f"hessian_eigenvalues={minimum.hessian_eigenvalues_v2_per_m4}, "
            f"optimizer_succeeded={minimum.optimizer_succeeded}"
        )


if __name__ == "__main__":
    main()
