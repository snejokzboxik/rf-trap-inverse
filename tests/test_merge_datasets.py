"""Focused tests for provenance-preserving clean-dataset merging."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from rf_trap_forward.merge_datasets import (
    MERGED_CLEAN_COLUMNS,
    MergeSource,
    merge_clean_datasets,
    write_merged_dataset,
)
from rf_trap_forward.synthetic_dataset import CLEAN_CSV_COLUMNS


def _row(sample_id: int, offset: int) -> dict[str, str]:
    """Create one exact-schema clean row with a distinct Wolfram input vector."""

    row = {column: "0" for column in CLEAN_CSV_COLUMNS}
    row.update(
        sample_id=str(sample_id), seed="7", status="clean",
        min_pairwise_distance_m="0.002", rejected_candidate_count="0",
    )
    for index, column in enumerate(
        ("w1_dx_m", "w1_dy_m", "w2_dx_m", "w2_dy_m", "w3_dx_m", "w3_dy_m", "w4_dx_m", "w4_dy_m"),
        start=1,
    ):
        row[column] = str((offset * 10 + index) * 1.0e-6)
    return row


def _source(directory: Path, seed: int, rows: list[dict[str, str]]) -> Path:
    """Write one tiny valid source directory without invoking FEM generation."""

    directory.mkdir()
    with (directory / "synthetic_clean.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CLEAN_CSV_COLUMNS)
        writer.writeheader(); writer.writerows(rows)
    (directory / "synthetic_summary.json").write_text(json.dumps({"seed": seed}), encoding="utf-8")
    return directory


def test_merge_adds_provenance_contiguous_ids_and_ml_view(tmp_path: Path) -> None:
    """Filtered sources must produce a duplicate-free exact-schema ML view."""

    first = _source(tmp_path / "first", 11, [_row(1, 1), _row(2, 2)])
    second = _source(tmp_path / "second", 22, [_row(1, 3), _row(2, 4)])
    result = merge_clean_datasets((MergeSource(first), MergeSource(second, min_sample_id=2)))
    assert [row["merged_sample_id"] for row in result.metadata_rows] == ["1", "2", "3"]
    assert [row["source_seed"] for row in result.metadata_rows] == ["11", "11", "22"]
    assert [row["source_sample_id"] for row in result.metadata_rows] == ["1", "2", "2"]
    assert result.duplicate_wolfram_inputs == result.duplicate_source_pairs == 0
    paths = write_merged_dataset(result, tmp_path / "merged")
    with paths["clean_csv"].open(encoding="utf-8", newline="") as stream:
        assert tuple(csv.DictReader(stream).fieldnames or ()) == MERGED_CLEAN_COLUMNS
        assert "sample_id" not in MERGED_CLEAN_COLUMNS
    with paths["ml_clean_csv"].open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
        assert tuple(rows[0]) == CLEAN_CSV_COLUMNS
    assert [row["sample_id"] for row in rows] == ["1", "2", "3"]
    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert summary["clean_samples"] == 3
    assert summary["wolfram_input_duplicates"] == 0
    assert summary["max_displacement_m"] == 500.0e-6
    assert summary["reference_row5_used"] is False


def test_duplicate_wolfram_input_is_rejected_before_writing(tmp_path: Path) -> None:
    """Two source rows with the same eight raw inputs must not merge silently."""

    first = _source(tmp_path / "first", 1, [_row(1, 1)])
    differently_formatted_duplicate = _row(1, 1)
    differently_formatted_duplicate["w1_dx_m"] = "0.000011"
    second = _source(tmp_path / "second", 2, [differently_formatted_duplicate])
    with pytest.raises(ValueError, match="duplicate Wolfram displacement input"):
        merge_clean_datasets((MergeSource(first), MergeSource(second)))


def test_merge_accepts_prior_ml_view_and_preserves_row_seeds(tmp_path: Path) -> None:
    """A previously merged ML view may carry multiple source seeds by row."""

    prior = tmp_path / "prior_merged"
    prior.mkdir()
    rows = [_row(1, 1), _row(2, 2)]
    rows[1]["seed"] = "8"
    with (prior / "synthetic_clean_ml.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=CLEAN_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (prior / "synthetic_summary.json").write_text(
        json.dumps({"kind": "merged_synthetic_dataset"}), encoding="utf-8"
    )

    result = merge_clean_datasets(
        (MergeSource(prior, clean_filename="synthetic_clean_ml.csv"),)
    )

    assert [row["source_seed"] for row in result.metadata_rows] == ["7", "8"]
    assert result.source_rows[0]["source_clean_file"] == "synthetic_clean_ml.csv"
    assert result.source_rows[0]["source_seeds"] == [7, 8]
