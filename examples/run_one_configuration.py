"""Run the milestone-one demonstrator for one asymmetric displacement vector."""

from __future__ import annotations

from rf_trap_forward import (
    run_forward_model,
)
from rf_trap_forward.demo import demonstrator_config, demonstrator_displacements_m


def main() -> None:
    """Execute the forward model and print its physical and numerical outputs."""

    result = run_forward_model(
        demonstrator_displacements_m(),
        demonstrator_config(),
    )

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
