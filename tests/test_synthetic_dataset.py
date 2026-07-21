"""Unit tests for bounded synthetic forward-dataset generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

import rf_trap_forward.synthetic_dataset as synthetic_dataset
from rf_trap_forward.absolute_validation import (
    wolfram_to_fem_absolute_displacements_m,
)
from rf_trap_forward.synthetic_dataset import (
    CLEAN_CSV_COLUMNS,
    REJECTED_CSV_COLUMNS,
    SyntheticDatasetConfig,
    SyntheticSolveResult,
    build_parser,
    generate_synthetic_dataset,
    generate_synthetic_dataset_incrementally,
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


def test_large_request_safety_gate_needs_no_fem_solve() -> None:
    """Only an explicit CLI/config acknowledgement permits more than 1000 rows."""

    assert SyntheticDatasetConfig(n=1000).n == 1000
    with pytest.raises(
        ValueError,
        match="n > 1000 requires --allow-large-n because generation may take many hours",
    ):
        SyntheticDatasetConfig(n=10_000)
    arguments = build_parser().parse_args(
        ("--n", "10000", "--allow-large-n", "--batch-size", "500", "--resume")
    )
    assert arguments.allow_large_n is True
    assert arguments.batch_size == 500
    assert arguments.resume is True
    assert SyntheticDatasetConfig(
        n=arguments.n,
        allow_large_n=arguments.allow_large_n,
    ).n == 10_000


def test_validated_large_request_reaches_sampling_without_fem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The acknowledged 10,000-row path must retain its large-run approval."""

    class SamplingReached(Exception):
        """Stop immediately after verifying the generator reaches sampling."""

    def stop_after_sampling(
        n: int,
        seed: int,
        max_displacement_m: float,
    ) -> np.ndarray:
        assert (n, seed, max_displacement_m) == (10_000, 7, 500.0e-6)
        raise SamplingReached

    monkeypatch.setattr(
        synthetic_dataset,
        "sample_wolfram_displacements_m",
        stop_after_sampling,
    )
    with pytest.raises(SamplingReached):
        synthetic_dataset.generate_synthetic_dataset(
            SyntheticDatasetConfig(n=10_000, seed=7, allow_large_n=True)
        )


def test_incremental_batches_create_files_and_flush_before_later_solves(
    tmp_path: Path,
) -> None:
    """A five-row mocked run must checkpoint each two-attempt batch durably."""

    output = tmp_path / "checkpointed"
    observed: list[int] = []

    def worker(*_: object) -> SyntheticSolveResult:
        assert output.is_dir()
        return _clean_worker()

    def progress(completed: int, _: int, __: float) -> None:
        observed.append(completed)
        with (output / "synthetic_clean.csv").open(encoding="utf-8", newline="") as stream:
            assert len(list(csv.DictReader(stream))) == completed
        checkpoint = json.loads((output / "progress.json").read_text(encoding="utf-8"))
        assert checkpoint["completed_attempts"] == completed

    result = generate_synthetic_dataset_incrementally(
        SyntheticDatasetConfig(n=5, seed=9, batch_size=2, max_workers=1),
        output,
        worker=worker,
        progress_callback=progress,
    )
    assert observed == [2, 4, 5]
    assert not result.interrupted
    assert result.progress.completed_attempts == 5
    assert result.progress.clean_count == 5
    assert result.paths.summary_json.is_file()


def test_keyboard_interrupt_writes_partial_files_and_resume_has_no_duplicates(
    tmp_path: Path,
) -> None:
    """A Ctrl+C-like worker interruption must save a usable contiguous prefix."""

    output = tmp_path / "interrupted"
    calls = 0

    def interrupting_worker(*_: object) -> SyntheticSolveResult:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return _clean_worker()

    config = SyntheticDatasetConfig(n=5, seed=10, batch_size=2, max_workers=1)
    partial = generate_synthetic_dataset_incrementally(
        config,
        output,
        worker=interrupting_worker,
    )
    assert partial.interrupted
    assert partial.progress.completed_attempts == 1
    progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "synthetic_summary.json").read_text(encoding="utf-8"))
    assert progress["last_sample_id"] == 1
    assert summary["partial"] is True

    resumed = generate_synthetic_dataset_incrementally(
        config,
        output,
        resume=True,
        worker=_clean_worker,
    )
    assert not resumed.interrupted
    with (output / "synthetic_clean.csv").open(encoding="utf-8", newline="") as stream:
        identifiers = [int(row["sample_id"]) for row in csv.DictReader(stream)]
    assert identifiers == [1, 2, 3, 4, 5]
    assert len(identifiers) == len(set(identifiers))


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
