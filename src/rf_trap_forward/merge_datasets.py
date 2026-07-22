"""Merge compatible clean synthetic datasets without changing their source files."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .synthetic_dataset import CLEAN_CSV_COLUMNS, REJECTED_CSV_COLUMNS


WOLFRAM_INPUT_COLUMNS = (
    "w1_dx_m", "w1_dy_m", "w2_dx_m", "w2_dy_m",
    "w3_dx_m", "w3_dy_m", "w4_dx_m", "w4_dy_m",
)
MERGED_METADATA_COLUMNS = (
    "merged_sample_id", "source_dataset", "source_seed", "source_sample_id",
)
MERGED_DATA_COLUMNS = tuple(
    column for column in CLEAN_CSV_COLUMNS if column != "sample_id"
)
MERGED_CLEAN_COLUMNS = MERGED_METADATA_COLUMNS + MERGED_DATA_COLUMNS


@dataclass(frozen=True)
class MergeSource:
    """One source dataset plus an optional inclusive original-ID filter."""

    directory: Path
    min_sample_id: int | None = None
    max_sample_id: int | None = None
    clean_filename: str = "synthetic_clean.csv"

    def __post_init__(self) -> None:
        """Reject an invalid requested source-ID interval."""

        if self.min_sample_id is not None and self.min_sample_id < 1:
            raise ValueError("min_sample_id must be positive")
        if (
            self.min_sample_id is not None
            and self.max_sample_id is not None
            and self.min_sample_id > self.max_sample_id
        ):
            raise ValueError("min_sample_id cannot exceed max_sample_id")
        if Path(self.clean_filename).name != self.clean_filename:
            raise ValueError("clean_filename must be a file name, not a path")


@dataclass(frozen=True)
class MergedDatasetResult:
    """Rows and evidence produced by one deterministic dataset merge."""

    sources: tuple[MergeSource, ...]
    metadata_rows: tuple[dict[str, str], ...]
    ml_rows: tuple[dict[str, str], ...]
    source_rows: tuple[dict[str, object], ...]
    duplicate_wolfram_inputs: int
    duplicate_source_pairs: int


def merge_clean_datasets(sources: Sequence[MergeSource]) -> MergedDatasetResult:
    """Read filtered clean CSVs, attach provenance, and reject duplicate inputs."""

    if not sources:
        raise ValueError("at least one source dataset is required")
    metadata_rows: list[dict[str, str]] = []
    ml_rows: list[dict[str, str]] = []
    source_rows: list[dict[str, object]] = []
    input_keys: set[tuple[float, ...]] = set()
    source_keys: set[tuple[str, int]] = set()
    duplicate_inputs = 0
    duplicate_pairs = 0
    for source in sources:
        clean_path = source.directory / source.clean_filename
        seed = _source_seed(source.directory)
        rows = _read_clean_rows(clean_path)
        included = 0
        included_seeds: set[int] = set()
        for row in rows:
            source_id = int(row["sample_id"])
            if source.min_sample_id is not None and source_id < source.min_sample_id:
                continue
            if source.max_sample_id is not None and source_id > source.max_sample_id:
                continue
            included += 1
            source_name = source.directory.name
            source_key = (source_name, source_id)
            if source_key in source_keys:
                duplicate_pairs += 1
            source_keys.add(source_key)
            input_key = tuple(float(row[column]) for column in WOLFRAM_INPUT_COLUMNS)
            if input_key in input_keys:
                duplicate_inputs += 1
            input_keys.add(input_key)
            merged_id = len(metadata_rows) + 1
            row_seed = _row_seed(row, seed)
            if row_seed is not None:
                included_seeds.add(row_seed)
            metadata = {
                "merged_sample_id": str(merged_id),
                "source_dataset": source_name,
                "source_seed": "" if row_seed is None else str(row_seed),
                "source_sample_id": str(source_id),
                **{column: row[column] for column in MERGED_DATA_COLUMNS},
            }
            metadata_rows.append(metadata)
            ml_row = {column: row[column] for column in CLEAN_CSV_COLUMNS}
            ml_row["sample_id"] = str(merged_id)
            ml_rows.append(ml_row)
        source_rows.append(
            {
                "source_dataset": source.directory.name,
                "source_directory": str(source.directory),
                "source_clean_file": source.clean_filename,
                "source_seed": seed,
                "source_seeds": sorted(included_seeds),
                "min_sample_id": source.min_sample_id,
                "max_sample_id": source.max_sample_id,
                "included_clean_rows": included,
            }
        )
    if duplicate_pairs:
        raise ValueError("duplicate source_dataset/source_sample_id pairs found")
    if duplicate_inputs:
        raise ValueError("duplicate Wolfram displacement input vectors found")
    return MergedDatasetResult(
        tuple(sources), tuple(metadata_rows), tuple(ml_rows), tuple(source_rows),
        duplicate_inputs, duplicate_pairs,
    )


def write_merged_dataset(
    result: MergedDatasetResult,
    output_directory: str | Path,
) -> dict[str, Path]:
    """Write provenance-rich and exact-schema ML views without source mutation."""

    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("merged output directory already contains files")
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "clean_csv": output / "synthetic_clean.csv",
        "ml_clean_csv": output / "synthetic_clean_ml.csv",
        "rejected_csv": output / "synthetic_rejected.csv",
        "summary_json": output / "synthetic_summary.json",
        "readme": output / "README.md",
    }
    _write_csv(paths["clean_csv"], MERGED_CLEAN_COLUMNS, result.metadata_rows)
    _write_csv(paths["ml_clean_csv"], CLEAN_CSV_COLUMNS, result.ml_rows)
    _write_csv(paths["rejected_csv"], REJECTED_CSV_COLUMNS, ())
    row_count = len(result.metadata_rows)
    summary = {
        "kind": "merged_synthetic_dataset",
        "clean_samples": row_count,
        "completed_samples": row_count,
        "requested_samples": row_count,
        "rejected_samples": 0,
        "ambiguous_branch_count": 0,
        "solver_failure_count": 0,
        "status_counts": {"clean": row_count},
        "ambiguous_minimum_distance_m": 0.00015,
        "coordinate_units": "metres",
        "max_displacement_m": 0.0005,
        "max_displacement_um": 500.0,
        "merged_sample_id_range": [1, row_count],
        "reference_row5_used": False,
        "source_dataset_source_sample_id_duplicates": result.duplicate_source_pairs,
        "wolfram_input_duplicates": result.duplicate_wolfram_inputs,
        "sources": list(result.source_rows),
        "wolfram_to_fem_transform": "[-W3, -W1, -W4, -W2]",
    }
    paths["summary_json"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths["readme"].write_text(_readme(summary), encoding="utf-8")
    return paths


def _read_clean_rows(path: Path) -> list[dict[str, str]]:
    """Load only a source clean CSV with the exact expected schema."""

    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != CLEAN_CSV_COLUMNS:
            raise ValueError(f"{path} does not have the expected clean CSV schema")
        rows = list(reader)
    if any(row.get("status") != "clean" for row in rows):
        raise ValueError(f"{path} contains a non-clean row")
    return rows


def _source_seed(directory: Path) -> int | None:
    """Read a source seed when its standard dataset summary is available."""

    summary = directory / "synthetic_summary.json"
    if not summary.is_file():
        return None
    value = json.loads(summary.read_text(encoding="utf-8")).get("seed")
    return int(value) if value is not None else None


def _row_seed(row: dict[str, str], fallback: int | None) -> int | None:
    """Use summary seed, or row seed for a mixed-seed prior merged ML view."""

    if fallback is not None:
        return fallback
    value = row.get("seed", "").strip()
    return int(value) if value else None


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    """Write a stable CSV schema and rows in deterministic merge order."""

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def _readme(summary: dict[str, object]) -> str:
    """Explain merged provenance and the metadata-free training view."""

    source_lines = []
    for source in summary["sources"]:
        source_lines.append(
            f"- `{source['source_dataset']}`: {source['included_clean_rows']} rows; "
            f"source sample IDs {source['min_sample_id'] or 'all'}..{source['max_sample_id'] or 'all'}; "
            f"input `{source['source_clean_file']}`; seeds {source['source_seeds']}."
        )
    return "\n".join((
        "# Merged synthetic RF-trap dataset", "",
        "This directory was created without modifying any source dataset.", "",
        "## Sources and filters", "", *source_lines, "",
        "## Files", "",
        "- `synthetic_clean.csv`: provenance-rich clean rows with merged and source IDs.",
        "- `synthetic_clean_ml.csv`: exact original clean schema, with `sample_id` replaced by contiguous `merged_sample_id`; use this for QA and inverse training.",
        "- `synthetic_rejected.csv`: empty exact-schema rejected view.",
        "- `synthetic_summary.json`: source/filter and duplicate-check evidence.", "",
        f"Total rows: **{summary['clean_samples']}**. Wolfram input duplicates: **{summary['wolfram_input_duplicates']}**. Source-pair duplicates: **{summary['source_dataset_source_sample_id_duplicates']}**.", "",
        "The physical convention remains `F1,F2,F3,F4 = -[W3,W1,W4,W2]`.", "",
    ))


def build_parser() -> argparse.ArgumentParser:
    """Build the multi-source merged-dataset CLI."""

    parser = argparse.ArgumentParser(
        prog="rf-trap-merge-datasets",
        description="Merge clean synthetic datasets with provenance and an ML view.",
    )
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument(
        "--clean-file", action="append", default=[],
        help="clean CSV file name per --source (default: synthetic_clean.csv)",
    )
    parser.add_argument(
        "--min-sample-id", type=int, action="append", default=[],
        help="optional inclusive minimum source ID per --source, in source order",
    )
    parser.add_argument(
        "--max-sample-id", type=int, action="append", default=[],
        help="optional inclusive maximum source ID per --source, in source order",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Merge requested source directories and print row-level verification facts."""

    arguments = build_parser().parse_args(argv)
    source_count = len(arguments.source)
    if len(arguments.min_sample_id) not in (0, source_count):
        raise ValueError("--min-sample-id must be omitted or supplied once per --source")
    if len(arguments.max_sample_id) not in (0, source_count):
        raise ValueError("--max-sample-id must be omitted or supplied once per --source")
    if len(arguments.clean_file) not in (0, source_count):
        raise ValueError("--clean-file must be omitted or supplied once per --source")
    minimums = arguments.min_sample_id or [None] * source_count
    maximums = arguments.max_sample_id or [None] * source_count
    clean_files = arguments.clean_file or ["synthetic_clean.csv"] * source_count
    result = merge_clean_datasets(
        tuple(
            MergeSource(path, minimum, maximum, clean_file)
            for path, minimum, maximum, clean_file in zip(
                arguments.source, minimums, maximums, clean_files, strict=True
            )
        )
    )
    paths = write_merged_dataset(result, arguments.output_dir)
    print(f"merged_clean_rows={len(result.metadata_rows)}")
    print(f"duplicate_wolfram_inputs={result.duplicate_wolfram_inputs}")
    print(f"duplicate_source_pairs={result.duplicate_source_pairs}")
    print(f"clean_csv={paths['clean_csv']}")
    print(f"ml_clean_csv={paths['ml_clean_csv']}")
    print(f"summary={paths['summary_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
