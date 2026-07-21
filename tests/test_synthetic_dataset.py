"""Unit tests for bounded synthetic forward-dataset generation."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from rf_trap_forward.absolute_validation import (
    wolfram_to_fem_absolute_displacements_m,
)
from rf_trap_forward.synthetic_dataset import (
    CLEAN_CSV_COLUMNS,
    REJECTED_CSV_COLUMNS,
    SyntheticDatasetConfig,
    SyntheticSolveResult,
    generate_synthetic_dataset,
    sample_wolfram_displacements_m,
    write_synthetic_dataset,
)


def _clean_worker(*_: object) -> SyntheticSolveResult:
    return SyntheticSolveResult(
        minima_positions_m=np.asarray(
            ((-2.0e-3, 0.0), (1.0e-3, 2.0e-3), (1.0e-3, -2.0e-3))
        ),
        accepted_candidate_count=3,
        rejected_candidate_count=2,
        total_candidate_count=5,
        selected_interpolation_sensitive_count=0,
        node_count=100,
        triangle_count=180,
        relative_free_residual=1.0e-15,
        runtime_seconds=0.1,
    )


def _ambiguous_worker(*_: object) -> SyntheticSolveResult:
    return SyntheticSolveResult(
        minima_positions_m=np.asarray(
            ((-2.0e-3, 0.0), (1.0e-3, 0.0), (1.05e-3, 0.0))
        ),
        accepted_candidate_count=3,
        rejected_candidate_count=0,
        total_candidate_count=3,
        selected_interpolation_sensitive_count=0,
        node_count=100,
        triangle_count=180,
        relative_free_residual=1.0e-15,
        runtime_seconds=0.1,
    )


def test_generator_wolfram_transform_is_exactly_signflip_perm3142() -> None:
    """Stored FEM inputs must be exactly ``[-W3,-W1,-W4,-W2]``."""

    result = generate_synthetic_dataset(
        SyntheticDatasetConfig(n=2, seed=11, batch_size=1),
        worker=_clean_worker,
    )
    for record in result.records:
        expected = wolfram_to_fem_absolute_displacements_m(
            record.wolfram_displacements_m
        )
        np.testing.assert_array_equal(record.fem_displacements_m, expected)


def test_sampling_is_deterministic_for_fixed_seed() -> None:
    """A fixed seed must reproduce every Wolfram-order displacement exactly."""

    first = sample_wolfram_displacements_m(4, 123, 500.0e-6)
    second = sample_wolfram_displacements_m(4, 123, 500.0e-6)
    different = sample_wolfram_displacements_m(4, 124, 500.0e-6)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)
    assert np.max(np.abs(first)) <= 500.0e-6


def test_csv_columns_are_stable_for_clean_and_rejected_files(
    tmp_path: Path,
) -> None:
    """Both split files must retain their documented machine-readable schemas."""

    clean = generate_synthetic_dataset(
        SyntheticDatasetConfig(n=1, seed=2, batch_size=1),
        worker=_clean_worker,
    )
    ambiguous = generate_synthetic_dataset(
        SyntheticDatasetConfig(n=1, seed=2, batch_size=1),
        worker=_ambiguous_worker,
    )
    clean_paths = write_synthetic_dataset(clean, tmp_path / "clean")
    rejected_paths = write_synthetic_dataset(ambiguous, tmp_path / "rejected")
    with clean_paths.clean_csv.open(encoding="utf-8", newline="") as stream:
        assert tuple(csv.DictReader(stream).fieldnames or ()) == CLEAN_CSV_COLUMNS
    with rejected_paths.rejected_csv.open(encoding="utf-8", newline="") as stream:
        assert tuple(csv.DictReader(stream).fieldnames or ()) == REJECTED_CSV_COLUMNS


def test_ambiguous_close_minima_are_excluded_from_clean_csv(
    tmp_path: Path,
) -> None:
    """A 50 µm minimum pair must be retained only as ``ambiguous_branch``."""

    result = generate_synthetic_dataset(
        SyntheticDatasetConfig(n=1, seed=3, batch_size=1),
        worker=_ambiguous_worker,
    )
    assert not result.clean_records
    assert result.rejected_records[0].status == "ambiguous_branch"
    paths = write_synthetic_dataset(result, tmp_path)
    with paths.clean_csv.open(encoding="utf-8", newline="") as stream:
        assert list(csv.DictReader(stream)) == []
    with paths.rejected_csv.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [item["status"] for item in rows] == ["ambiguous_branch"]


def test_more_than_three_accepted_candidates_is_not_clean() -> None:
    """Lowest-three selection must not hide a fourth robust-accepted candidate."""

    def extra_candidate_worker(*_: object) -> SyntheticSolveResult:
        base = _clean_worker()
        return SyntheticSolveResult(
            minima_positions_m=base.minima_positions_m,
            accepted_candidate_count=4,
            rejected_candidate_count=1,
            total_candidate_count=5,
            selected_interpolation_sensitive_count=0,
            node_count=base.node_count,
            triangle_count=base.triangle_count,
            relative_free_residual=base.relative_free_residual,
            runtime_seconds=base.runtime_seconds,
        )

    result = generate_synthetic_dataset(
        SyntheticDatasetConfig(n=1, seed=4, batch_size=1),
        worker=extra_candidate_worker,
    )
    assert not result.clean_records
    assert result.rejected_records[0].status == "not_exactly_three_robust_minima"


def test_tiny_two_sample_generation_runs_without_crashing() -> None:
    """The complete generator orchestration must handle a two-row request."""

    result = generate_synthetic_dataset(
        SyntheticDatasetConfig(n=2, seed=123, batch_size=2),
        worker=_clean_worker,
    )
    assert len(result.records) == 2
    assert len(result.clean_records) == 2
    assert not result.rejected_records
    assert [item.sample_id for item in result.records] == [1, 2]
