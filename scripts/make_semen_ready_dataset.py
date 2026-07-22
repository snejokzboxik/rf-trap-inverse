"""Build the minimal 8-6-8 FEM-order CSV requested by Semen."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Sequence


SOURCE_CSV = Path(
    "validation_results/prediction_export_merged_51974/prediction_dataset_300.csv"
)
OUTPUT_DIR = Path("validation_results/semen_ready_dataset_100")
OUTPUT_CSV_NAME = "semen_ready_100_fem_order.csv"
README_NAME = "semen_ready_100_readme.md"
SUMMARY_NAME = "semen_ready_100_summary.json"
ROW_COUNT = 100

TRUE_W_COLUMNS = tuple(
    f"true_w{electrode}_{component}_m"
    for electrode in range(1, 5)
    for component in ("dx", "dy")
)
PRED_W_COLUMNS = tuple(
    f"pred_w{electrode}_{component}_m"
    for electrode in range(1, 5)
    for component in ("dx", "dy")
)
MINIMA_COLUMNS = tuple(
    f"min{minimum}_{component}_m"
    for minimum in range(1, 4)
    for component in ("x", "y")
)
TRUE_F_COLUMNS = tuple(
    f"true_f{electrode}_{component}_m"
    for electrode in range(1, 5)
    for component in ("dx", "dy")
)
PRED_F_COLUMNS = tuple(
    f"pred_f{electrode}_{component}_m"
    for electrode in range(1, 5)
    for component in ("dx", "dy")
)
OUTPUT_COLUMNS = (*TRUE_F_COLUMNS, *MINIMA_COLUMNS, *PRED_F_COLUMNS)


def wolfram_flat_to_fem_flat(values: Sequence[float]) -> tuple[float, ...]:
    """Apply ``F1,F2,F3,F4 = -[W3,W1,W4,W2]`` to eight coordinates."""

    if len(values) != 8:
        raise ValueError("Wolfram displacement vector must contain eight coordinates")
    pairs = tuple((float(values[index]), float(values[index + 1])) for index in range(0, 8, 2))
    if not all(math.isfinite(value) for pair in pairs for value in pair):
        raise ValueError("Wolfram displacement coordinates must be finite")
    fem_pairs = tuple((-pairs[index][0], -pairs[index][1]) for index in (2, 0, 3, 1))
    return tuple(value for pair in fem_pairs for value in pair)


def _numeric_values(row: dict[str, str], columns: Sequence[str]) -> tuple[float, ...]:
    try:
        values = tuple(float(row[column]) for column in columns)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("source CSV contains a missing or malformed numeric value") from error
    if not all(math.isfinite(value) for value in values):
        raise ValueError("source CSV contains NaN or infinite values")
    return values


def build_rows(source_csv: Path, row_count: int = ROW_COUNT) -> list[dict[str, float]]:
    """Read the first rows and return the exact 22-column FEM-order records."""

    if row_count <= 0:
        raise ValueError("row_count must be positive")
    with source_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {*TRUE_W_COLUMNS, *MINIMA_COLUMNS, *PRED_W_COLUMNS}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required.difference(reader.fieldnames or ()))
            raise ValueError(f"source CSV is missing required columns: {missing}")
        source_rows = []
        for source_row in reader:
            source_rows.append(source_row)
            if len(source_rows) == row_count:
                break
    if len(source_rows) != row_count:
        raise ValueError(
            f"source CSV contains only {len(source_rows)} rows; {row_count} required"
        )

    output_rows: list[dict[str, float]] = []
    for source_row in source_rows:
        true_w = _numeric_values(source_row, TRUE_W_COLUMNS)
        minima = _numeric_values(source_row, MINIMA_COLUMNS)
        pred_w = _numeric_values(source_row, PRED_W_COLUMNS)
        true_f = wolfram_flat_to_fem_flat(true_w)
        pred_f = wolfram_flat_to_fem_flat(pred_w)
        output: dict[str, float] = {}
        output.update(zip(TRUE_F_COLUMNS, true_f, strict=True))
        output.update(zip(MINIMA_COLUMNS, minima, strict=True))
        output.update(zip(PRED_F_COLUMNS, pred_f, strict=True))
        output_rows.append(output)
    return output_rows


def validate_rows(rows: Sequence[dict[str, float]], row_count: int = ROW_COUNT) -> None:
    """Assert exact shape, numeric finiteness, and plausible metre-scale values."""

    if len(rows) != row_count:
        raise AssertionError(f"expected {row_count} rows, found {len(rows)}")
    if len(OUTPUT_COLUMNS) != 22:
        raise AssertionError(f"expected 22 columns, found {len(OUTPUT_COLUMNS)}")
    displacement_columns = (*TRUE_F_COLUMNS, *PRED_F_COLUMNS)
    for row_index, row in enumerate(rows):
        if tuple(row) != OUTPUT_COLUMNS:
            raise AssertionError(f"row {row_index} does not use the exact output schema")
        values = tuple(row[column] for column in OUTPUT_COLUMNS)
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            raise AssertionError(f"row {row_index} contains a non-finite numeric value")
        if any(abs(row[column]) >= 0.01 for column in displacement_columns):
            raise AssertionError(
                f"row {row_index} has a displacement inconsistent with metre units"
            )


def write_outputs(
    rows: Sequence[dict[str, float]],
    *,
    source_csv: Path,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """Write the minimal CSV and its concise usage documentation."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / OUTPUT_CSV_NAME
    readme = output_dir / README_NAME
    summary = output_dir / SUMMARY_NAME

    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(OUTPUT_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    displacement_values = [
        row[column]
        for row in rows
        for column in (*TRUE_F_COLUMNS, *PRED_F_COLUMNS)
    ]
    summary_record = {
        "column_count": len(OUTPUT_COLUMNS),
        "column_order": list(OUTPUT_COLUMNS),
        "direct_fem_substitution": True,
        "finite_values": True,
        "minima_order": "copied unchanged; canonical atan2 order from source",
        "output_csv": str(output_csv),
        "row_count": len(rows),
        "selection": "first 100 source rows; no randomization",
        "source_csv": str(source_csv),
        "transform_applied_to_true_and_predicted": "F1,F2,F3,F4 = -[W3,W1,W4,W2]",
        "units": "metres",
        "maximum_absolute_displacement_m": max(abs(value) for value in displacement_values),
    }
    summary.write_text(
        json.dumps(summary_record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    readme.write_text(
        "\n".join(
            (
                "# Semen-ready FEM-order prediction dataset",
                "",
                "This file is FEM-order only and is ready for direct FEM substitution.",
                "Do not apply the Wolfram-to-FEM transform again.",
                "",
                "- Rows: 100 (the first 100 rows of the source export; no randomization).",
                "- Columns: 22 numeric columns only.",
                "- All values are in metres.",
                "- First 8 columns: true FEM displacements in F1, F2, F3, F4 order.",
                "- Middle 6 columns: the three equilibrium minima, copied unchanged in canonical atan2 order.",
                "- Last 8 columns: predicted FEM displacements in F1, F2, F3, F4 order.",
                "",
                "Both displacement blocks already use:",
                "",
                "`F1 = -W3, F2 = -W1, F3 = -W4, F4 = -W2`",
                "",
                f"Source: `{source_csv.as_posix()}`",
                "",
            )
        ),
        encoding="utf-8",
    )
    return output_csv, readme, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, default=SOURCE_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--rows", type=int, default=ROW_COUNT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    rows = build_rows(arguments.source_csv, arguments.rows)
    validate_rows(rows, arguments.rows)
    output_csv, readme, summary = write_outputs(
        rows, source_csv=arguments.source_csv, output_dir=arguments.output_dir
    )
    print(f"output_csv={output_csv}")
    print(f"readme={readme}")
    print(f"summary={summary}")
    print(f"shape={len(rows)}x{len(OUTPUT_COLUMNS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
