"""Command-line entry points for forward-model validation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .demo import demonstrator_config, demonstrator_displacements_m
from .validation import (
    ConvergenceStudyConfig,
    run_convergence_study,
    write_convergence_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the convergence-study command-line argument parser."""

    parser = argparse.ArgumentParser(
        prog="rf-trap-convergence",
        description=(
            "Run the provisional displacement configuration over a full "
            "mesh-size by outer-radius convergence grid."
        ),
    )
    parser.add_argument(
        "--mesh-sizes-um",
        type=float,
        nargs="+",
        default=(120.0, 80.0, 60.0),
        help="mesh characteristic lengths in micrometres (at least two)",
    )
    parser.add_argument(
        "--outer-radii-mm",
        type=float,
        nargs="+",
        default=(3.5, 4.0, 5.0),
        help="circular outer-boundary radii in millimetres (at least two)",
    )
    parser.add_argument(
        "--coordinate-tolerance-um",
        type=float,
        default=10.0,
        help="reported acceptance tolerance for every successive coordinate change",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("validation_results") / "milestone_2",
        help="directory receiving CSV, Markdown, and PNG outputs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the demonstrator convergence study and write all report artifacts."""

    arguments = build_parser().parse_args(argv)
    study_config = ConvergenceStudyConfig(
        mesh_sizes_m=tuple(value * 1.0e-6 for value in arguments.mesh_sizes_um),
        outer_radii_m=tuple(value * 1.0e-3 for value in arguments.outer_radii_mm),
        coordinate_tolerance_m=arguments.coordinate_tolerance_um * 1.0e-6,
    )
    report = run_convergence_study(
        demonstrator_displacements_m(),
        demonstrator_config(),
        study_config,
    )
    paths = write_convergence_outputs(report, arguments.output_directory)
    print(f"runs: {len(report.runs)}")
    print(
        "three-minimum structure stable: "
        f"{report.three_minimum_structure_stable}"
    )
    print(
        "maximum mesh-refinement shift (um): "
        f"{report.maximum_coordinate_change_m('mesh_size_m') * 1.0e6:.6g}"
    )
    print(
        "maximum outer-radius shift (um): "
        f"{report.maximum_coordinate_change_m('outer_radius_m') * 1.0e6:.6g}"
    )
    print(f"markdown report: {paths.markdown_report}")
    return 0 if report.three_minimum_structure_stable else 2


if __name__ == "__main__":
    raise SystemExit(main())

