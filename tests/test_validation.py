"""Synthetic and mocked tests for convergence reporting utilities."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rf_trap_forward import ForwardModelConfig, MeshConfig
from rf_trap_forward.minima import LocalMinimum, MinimaDiagnostics
from rf_trap_forward.validation import (
    ConvergenceRunRecord,
    ConvergenceStudyConfig,
    build_convergence_report,
    compare_successive_minima,
    run_convergence_study,
    write_convergence_outputs,
)


def _minimum(x_m: float, y_m: float) -> LocalMinimum:
    return LocalMinimum(
        position_m=np.asarray([x_m, y_m], dtype=float),
        pseudopotential_v2_per_m2=1.0e-4,
        hessian_eigenvalues_v2_per_m4=np.asarray([1.0e10, 2.0e10]),
        optimizer_succeeded=True,
    )


def _record(
    mesh_size_m: float,
    outer_radius_m: float,
    positions_m: tuple[tuple[float, float], ...] | None = None,
) -> ConvergenceRunRecord:
    positions = positions_m or (
        (0.30e-3, 0.25e-3),
        (-0.47e-3, 0.10e-3),
        (0.32e-3, -0.22e-3),
    )
    diagnostics = MinimaDiagnostics(
        valid_coarse_points=100,
        coarse_candidates=len(positions),
        refined_candidates=len(positions),
        unique_candidates=len(positions),
        hessian_validated_candidates=len(positions),
    )
    return ConvergenceRunRecord(
        mesh_size_m=mesh_size_m,
        boundary_tolerance_m=1.0e-9,
        gmsh_algorithm=6,
        random_seed=1,
        random_factor=0.0,
        gmsh_reproducible=True,
        outer_radius_m=outer_radius_m,
        node_count=1000,
        triangle_count=1900,
        relative_free_residual=1.0e-14,
        electrode_boundary_error_v=0.0,
        outer_boundary_error_v=0.0,
        minima_diagnostics=diagnostics,
        minima=tuple(_minimum(*position) for position in positions),
    )


def _study_config() -> ConvergenceStudyConfig:
    return ConvergenceStudyConfig(
        mesh_sizes_m=(0.12e-3, 0.08e-3),
        outer_radii_m=(3.5e-3, 4.0e-3),
        coordinate_tolerance_m=20.0e-6,
    )


def _matrix_records() -> tuple[ConvergenceRunRecord, ...]:
    records = []
    for radius in (3.5e-3, 4.0e-3):
        records.append(_record(0.12e-3, radius))
        shifted = (
            (0.301e-3, 0.249e-3),
            (-0.469e-3, 0.101e-3),
            (0.319e-3, -0.219e-3),
        )
        records.append(_record(0.08e-3, radius, shifted))
    return tuple(records)


def test_successive_comparison_uses_spatial_assignment() -> None:
    """Reordered minima must be paired by proximity rather than tuple position."""

    previous = _record(
        0.12e-3,
        4.0e-3,
        ((1.0e-3, 0.0), (-1.0e-3, 0.0), (0.0, 1.0e-3)),
    )
    current = _record(
        0.08e-3,
        4.0e-3,
        ((-0.99e-3, 0.0), (0.0, 1.01e-3), (1.01e-3, 0.0)),
    )
    comparisons = compare_successive_minima((previous, current))
    assert len(comparisons) == 3
    np.testing.assert_allclose(
        sorted(item.distance_m for item in comparisons),
        [10.0e-6, 10.0e-6, 10.0e-6],
    )


def test_report_separates_structure_stability_from_coordinate_tolerance() -> None:
    """Topology and a user-selected coordinate tolerance are independent decisions."""

    report = build_convergence_report(
        np.zeros(6),
        _study_config(),
        _matrix_records(),
    )
    assert report.three_minimum_structure_stable
    assert report.coordinate_changes_within_tolerance
    assert len(report.comparisons) == 12

    incomplete = list(_matrix_records())
    incomplete[0] = _record(
        0.12e-3,
        3.5e-3,
        ((0.30e-3, 0.25e-3), (-0.47e-3, 0.10e-3)),
    )
    unstable = build_convergence_report(np.zeros(6), _study_config(), incomplete)
    assert not unstable.three_minimum_structure_stable

    extra_candidate_records = list(_matrix_records())
    first = extra_candidate_records[0]
    extra_candidate_records[0] = replace(
        first,
        minima_diagnostics=MinimaDiagnostics(
            valid_coarse_points=100,
            coarse_candidates=4,
            refined_candidates=4,
            unique_candidates=4,
            hessian_validated_candidates=4,
        ),
    )
    extra_candidate_report = build_convergence_report(
        np.zeros(6),
        _study_config(),
        extra_candidate_records,
    )
    assert not extra_candidate_report.three_minimum_structure_stable


def test_full_factorial_study_uses_mocked_forward_runner(
    model_config: ForwardModelConfig,
) -> None:
    """Every requested mesh/radius pair must reach the injected forward runner."""

    calls: list[tuple[float, float]] = []

    def fake_runner(displacements_m, config: ForwardModelConfig):
        calls.append(
            (
                config.mesh.characteristic_length_m,
                config.geometry.outer_radius_m,
            )
        )
        record = _record(
            config.mesh.characteristic_length_m,
            config.geometry.outer_radius_m,
        )
        return SimpleNamespace(
            geometry=SimpleNamespace(config=config.geometry),
            trap_mesh=SimpleNamespace(
                number_of_nodes=record.node_count,
                number_of_triangles=record.triangle_count,
            ),
            fem_solution=SimpleNamespace(
                relative_free_residual=record.relative_free_residual,
                electrode_boundary_error_v=record.electrode_boundary_error_v,
                outer_boundary_error_v=record.outer_boundary_error_v,
            ),
            minima_diagnostics=record.minima_diagnostics,
            minima=record.minima,
        )

    report = run_convergence_study(
        np.zeros(6),
        model_config,
        _study_config(),
        runner=fake_runner,
    )
    assert len(calls) == 4
    assert set(calls) == {
        (0.12e-3, 3.5e-3),
        (0.08e-3, 3.5e-3),
        (0.12e-3, 4.0e-3),
        (0.08e-3, 4.0e-3),
    }
    assert len(report.runs) == 4


def test_output_writer_creates_complete_tables_and_headless_plots(tmp_path: Path) -> None:
    """Synthetic records must produce readable CSV, Markdown, and PNG artifacts."""

    report = build_convergence_report(
        np.zeros(6),
        _study_config(),
        _matrix_records(),
    )
    paths = write_convergence_outputs(report, tmp_path)

    with paths.runs_csv.open(encoding="utf-8", newline="") as stream:
        run_rows = list(csv.DictReader(stream))
    with paths.comparisons_csv.open(encoding="utf-8", newline="") as stream:
        comparison_rows = list(csv.DictReader(stream))
    markdown = paths.markdown_report.read_text(encoding="utf-8")

    assert len(run_rows) == 4
    assert len(comparison_rows) == 12
    assert "minimum_3_hessian_eigenvalue_2_v2_per_m4" in run_rows[0]
    assert "Three-minimum structure stable: **yes**" in markdown
    assert paths.mesh_refinement_plot.read_bytes().startswith(b"\x89PNG")
    assert paths.outer_radius_plot.read_bytes().startswith(b"\x89PNG")


def test_study_config_rejects_singleton_axes() -> None:
    """A convergence axis with no successive comparison must be rejected."""

    with pytest.raises(ValueError, match="at least two"):
        ConvergenceStudyConfig(
            mesh_sizes_m=(0.08e-3,),
            outer_radii_m=(3.5e-3, 4.0e-3),
        )


def test_mesh_config_rejects_zero_random_seed() -> None:
    """Gmsh seed zero is rejected because it does not reliably reset run history."""

    with pytest.raises(ValueError, match="must be positive"):
        MeshConfig(characteristic_length_m=0.08e-3, random_seed=0)
