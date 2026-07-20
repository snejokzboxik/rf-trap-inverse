"""Tests for FEM-to-reference benchmark logic without expensive FEM solves."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rf_trap_forward.dataset import ReferenceDataset
from rf_trap_forward.demo import demonstrator_config
from rf_trap_forward.reference_validation import (
    ForwardObservation,
    match_minima_by_distance,
    run_reference_validation,
    select_reference_rows,
    write_reference_validation_outputs,
)


class _MockForwardResult:
    """Minimal object exposing the forward-result interface used by validation."""

    def __init__(
        self,
        positions_m: np.ndarray,
        *,
        hessian_validated_candidates: int = 3,
    ) -> None:
        self._positions_m = positions_m
        self.minima_diagnostics = SimpleNamespace(
            hessian_validated_candidates=hessian_validated_candidates,
            valid_coarse_points=100,
            coarse_candidates=3,
            refined_candidates=3,
            unique_candidates=3,
        )
        self.trap_mesh = SimpleNamespace(number_of_nodes=50, number_of_triangles=80)
        self.fem_solution = SimpleNamespace(relative_free_residual=2.0e-15)

    def minima_positions_m(self) -> np.ndarray:
        """Return the mocked angle-ordered positions."""

        return self._positions_m.copy()


def _one_row_dataset() -> ReferenceDataset:
    raw_displacements = np.asarray(
        [[[1.0, 2.0], [4.0, 8.0], [-2.0, 5.0], [7.0, -1.0]]],
        dtype=float,
    ) * 1.0e-4
    raw_minima = np.asarray(
        [[[2.0, 2.0], [1.0, 4.0], [-1.0, 2.0]]],
        dtype=float,
    ) * 1.0e-3
    return ReferenceDataset(raw_displacements, raw_minima)


def test_row_selection_defaults_range_and_random_subset() -> None:
    """Selection must default to ten and support inclusive deterministic sampling."""

    assert select_reference_rows(326) == tuple(range(1, 11))
    assert select_reference_rows(326, start_row=8, end_row=12) == (8, 9, 10, 11, 12)
    first = select_reference_rows(326, random_count=7, random_seed=19)
    second = select_reference_rows(326, random_count=7, random_seed=19)
    assert first == second
    assert len(first) == 7
    assert all(1 <= row <= 326 for row in first)
    assert select_reference_rows(
        326,
        start_row=20,
        end_row=40,
        random_count=4,
        random_seed=3,
    ) == tuple(sorted(select_reference_rows(
        326,
        start_row=20,
        end_row=40,
        random_count=4,
        random_seed=3,
    )))


def test_minimum_distance_assignment_ignores_input_order() -> None:
    """Spatial assignment must recover the nearest permutation, not polar indices."""

    reference = np.asarray([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]])
    computed = np.asarray([[0.1, 3.0], [0.2, 0.0], [3.8, 0.0]])
    matches = match_minima_by_distance(reference, computed)
    assert [match.computed_index for match in matches] == [2, 3, 1]
    np.testing.assert_allclose(
        [match.distance_m for match in matches],
        [0.2, 0.2, 0.1],
    )


def test_mocked_forward_run_uses_electrode1_relative_convention() -> None:
    """The runner must receive 6D relative inputs and compare translated minima."""

    dataset = _one_row_dataset()
    reference_relative = dataset.minima_relative_to_electrode1_m[0]
    computed = reference_relative[[2, 0, 1]] + np.asarray([2.0e-6, -1.0e-6])
    captured: list[np.ndarray] = []

    def runner(displacements_m: object, _config: object) -> _MockForwardResult:
        captured.append(np.asarray(displacements_m, dtype=float))
        return _MockForwardResult(computed)

    report = run_reference_validation(
        dataset,
        demonstrator_config(),
        (1,),
        runner=runner,
    )
    np.testing.assert_allclose(
        captured[0],
        np.asarray([3.0, 6.0, -3.0, 3.0, 6.0, -3.0]) * 1.0e-4,
    )
    row = report.rows[0]
    assert row.completed
    assert row.exactly_three_physical_minima
    assert [match.computed_index for match in row.matches] == [2, 3, 1]
    np.testing.assert_allclose(row.error_distances_m(), np.sqrt(5.0) * 1.0e-6)


def test_summary_metrics_include_all_matched_minima() -> None:
    """Aggregate mean, median, maximum, and p95 must use matched distances."""

    dataset = _one_row_dataset()
    reference = dataset.minima_relative_to_electrode1_m[0]
    offsets = np.asarray([[1.0e-6, 0.0], [0.0, 2.0e-6], [4.0e-6, 0.0]])
    observation = ForwardObservation(
        minima_positions_m=reference + offsets,
        hessian_validated_candidates=4,
        node_count=10,
        triangle_count=12,
        relative_free_residual=1.0e-15,
        valid_coarse_points=20,
        coarse_candidates=4,
        refined_candidates=4,
        unique_candidates=4,
    )
    report = run_reference_validation(
        dataset,
        demonstrator_config(),
        (1,),
        runner=lambda _displacements, _config: observation,
    )
    summary = report.summary()
    assert summary.completed_rows == 1
    assert summary.rows_with_exactly_three_physical_minima == 0
    assert summary.mean_error_m == pytest.approx(np.mean([1.0e-6, 2.0e-6, 4.0e-6]))
    assert summary.median_error_m == pytest.approx(2.0e-6)
    assert summary.maximum_error_m == pytest.approx(4.0e-6)
    assert summary.percentile_95_error_m == pytest.approx(
        np.percentile([1.0e-6, 2.0e-6, 4.0e-6], 95.0)
    )


def test_failed_mocked_forward_row_is_retained() -> None:
    """A solver error must produce a report row instead of aborting the study."""

    def runner(_displacements: object, _config: object) -> _MockForwardResult:
        raise ValueError("electrode disks must not touch or overlap")

    report = run_reference_validation(
        _one_row_dataset(),
        demonstrator_config(),
        (1,),
        runner=runner,
    )
    row = report.rows[0]
    assert row.status == "forward-failed"
    assert row.error_type == "ValueError"
    assert not row.matches
    assert report.summary().matched_minima == 0


def test_csv_markdown_and_plot_outputs_from_mocked_data(tmp_path: Path) -> None:
    """All report artifacts must be generated from inexpensive mocked data."""

    dataset = _one_row_dataset()
    reference = dataset.minima_relative_to_electrode1_m[0]
    report = run_reference_validation(
        dataset,
        demonstrator_config(),
        (1,),
        runner=lambda _displacements, _config: _MockForwardResult(
            reference + np.asarray([1.0e-6, 0.0])
        ),
    )
    paths = write_reference_validation_outputs(report, tmp_path / "report")

    with paths.rows_csv.open(encoding="utf-8", newline="") as stream:
        row_records = list(csv.DictReader(stream))
    with paths.minima_csv.open(encoding="utf-8", newline="") as stream:
        minimum_records = list(csv.DictReader(stream))
    assert len(row_records) == 1
    assert float(row_records[0]["mean_error_um"]) == pytest.approx(1.0)
    assert len(minimum_records) == 3
    assert all(
        float(record["error_mm"]) == pytest.approx(0.001)
        for record in minimum_records
    )
    markdown = paths.markdown_report.read_text(encoding="utf-8")
    assert "minimum-total-distance assignment" in markdown
    assert "Rows with exactly three" in markdown
    assert len(paths.plot_paths) == 1
    assert paths.plot_paths[0].stat().st_size > 1_000
