"""Build a minimal direct-FEM-order 8-6-8 prediction dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from numpy.typing import NDArray


DEFAULT_DATASET = Path(
    "validation_results/generated_dataset_merged_51974/synthetic_clean_ml.csv"
)
DEFAULT_MODEL = Path("validation_results/inverse_model_merged_51974/mlp.joblib")
DEFAULT_OUTPUT_DIR = Path("validation_results/semen_ready_dataset_100")
DEFAULT_N = 100

SOURCE_W_COLUMNS = tuple(
    f"w{electrode}_{component}_m"
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


@dataclass(frozen=True)
class FEMOrderExport:
    """Source values, model predictions, and exact 22-column output rows."""

    true_w_m: NDArray[np.float64]
    minima_m: NDArray[np.float64]
    predicted_w_m: NDArray[np.float64]
    rows: tuple[dict[str, float], ...]


def wolfram_flat_to_fem_flat(values: Sequence[float]) -> tuple[float, ...]:
    """Apply ``F1,F2,F3,F4 = -[W3,W1,W4,W2]`` to eight coordinates."""

    if len(values) != 8:
        raise ValueError("Wolfram displacement vector must contain eight coordinates")
    pairs = tuple(
        (float(values[index]), float(values[index + 1]))
        for index in range(0, 8, 2)
    )
    if not all(math.isfinite(value) for pair in pairs for value in pair):
        raise ValueError("Wolfram displacement coordinates must be finite")
    fem_pairs = tuple(
        (-pairs[index][0], -pairs[index][1]) for index in (2, 0, 3, 1)
    )
    return tuple(value for pair in fem_pairs for value in pair)


def _numeric_values(
    row: dict[str, str], columns: Sequence[str]
) -> tuple[float, ...]:
    try:
        values = tuple(float(row[column]) for column in columns)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "source CSV contains a missing or malformed numeric value"
        ) from error
    if not all(math.isfinite(value) for value in values):
        raise ValueError("source CSV contains NaN or infinite values")
    return values


def _load_first_rows(
    dataset_csv: Path, row_count: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read the first clean rows without changing their canonical minima order."""

    if row_count <= 0:
        raise ValueError("n must be positive")
    with dataset_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {*SOURCE_W_COLUMNS, *MINIMA_COLUMNS}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required.difference(reader.fieldnames or ()))
            raise ValueError(f"source CSV is missing required columns: {missing}")
        true_w_rows: list[tuple[float, ...]] = []
        minima_rows: list[tuple[float, ...]] = []
        for source_row in reader:
            true_w_rows.append(_numeric_values(source_row, SOURCE_W_COLUMNS))
            minima_rows.append(_numeric_values(source_row, MINIMA_COLUMNS))
            if len(true_w_rows) == row_count:
                break
    if len(true_w_rows) != row_count:
        raise ValueError(
            f"source CSV contains only {len(true_w_rows)} rows; {row_count} required"
        )
    return (
        np.asarray(true_w_rows, dtype=float),
        np.asarray(minima_rows, dtype=float),
    )


def build_export(
    dataset_csv: Path,
    model_path: Path,
    row_count: int = DEFAULT_N,
) -> FEMOrderExport:
    """Predict the first rows and construct direct-FEM-order output records."""

    true_w_m, minima_m = _load_first_rows(dataset_csv, row_count)
    model = joblib.load(model_path)
    if not callable(getattr(model, "predict", None)):
        raise ValueError("loaded inverse model does not provide predict")
    predicted_w_m = np.asarray(model.predict(minima_m), dtype=float)
    if predicted_w_m.shape != (row_count, 8):
        raise ValueError(
            "model predictions must have shape "
            f"{(row_count, 8)}, got {predicted_w_m.shape}"
        )
    if not np.all(np.isfinite(predicted_w_m)):
        raise ValueError("model predictions contain NaN or infinite values")

    output_rows: list[dict[str, float]] = []
    for row_index in range(row_count):
        true_f = wolfram_flat_to_fem_flat(true_w_m[row_index])
        predicted_f = wolfram_flat_to_fem_flat(predicted_w_m[row_index])
        output: dict[str, float] = {}
        output.update(zip(TRUE_F_COLUMNS, true_f, strict=True))
        output.update(zip(MINIMA_COLUMNS, minima_m[row_index], strict=True))
        output.update(zip(PRED_F_COLUMNS, predicted_f, strict=True))
        output_rows.append(output)
    return FEMOrderExport(
        true_w_m=true_w_m,
        minima_m=minima_m,
        predicted_w_m=predicted_w_m,
        rows=tuple(output_rows),
    )


def validate_export(export: FEMOrderExport, row_count: int = DEFAULT_N) -> None:
    """Assert exact shape, numeric finiteness, units, and both transforms."""

    if len(export.rows) != row_count:
        raise AssertionError(
            f"expected {row_count} rows, found {len(export.rows)}"
        )
    if len(OUTPUT_COLUMNS) != 22:
        raise AssertionError(f"expected 22 columns, found {len(OUTPUT_COLUMNS)}")
    if any("w1_" in column or "w2_" in column or "w3_" in column or "w4_" in column for column in OUTPUT_COLUMNS):
        raise AssertionError("Wolfram-order columns must not appear in the final CSV")

    displacement_columns = (*TRUE_F_COLUMNS, *PRED_F_COLUMNS)
    for row_index, row in enumerate(export.rows):
        if tuple(row) != OUTPUT_COLUMNS:
            raise AssertionError(
                f"row {row_index} does not use the exact output schema"
            )
        values = tuple(row[column] for column in OUTPUT_COLUMNS)
        if not all(
            isinstance(value, (int, float, np.floating))
            and math.isfinite(float(value))
            for value in values
        ):
            raise AssertionError(
                f"row {row_index} contains a non-finite numeric value"
            )
        if any(abs(float(row[column])) >= 0.01 for column in displacement_columns):
            raise AssertionError(
                f"row {row_index} has a displacement inconsistent with metre units"
            )
        expected_true_f = wolfram_flat_to_fem_flat(export.true_w_m[row_index])
        expected_predicted_f = wolfram_flat_to_fem_flat(
            export.predicted_w_m[row_index]
        )
        observed_true_f = tuple(float(row[column]) for column in TRUE_F_COLUMNS)
        observed_predicted_f = tuple(
            float(row[column]) for column in PRED_F_COLUMNS
        )
        observed_minima = tuple(float(row[column]) for column in MINIMA_COLUMNS)
        if observed_true_f != expected_true_f:
            raise AssertionError(
                f"row {row_index} true FEM transform is incorrect"
            )
        if observed_predicted_f != expected_predicted_f:
            raise AssertionError(
                f"row {row_index} predicted FEM transform is incorrect"
            )
        if observed_minima != tuple(float(value) for value in export.minima_m[row_index]):
            raise AssertionError(
                f"row {row_index} minima differ from the source dataset"
            )


def write_outputs(
    export: FEMOrderExport,
    *,
    dataset_csv: Path,
    model_path: Path,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """Write the exact CSV plus neutral usage documentation and summary."""

    row_count = len(export.rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"semen_ready_{row_count}_fem_order.csv"
    readme = output_dir / f"semen_ready_{row_count}_readme.md"
    summary = output_dir / f"semen_ready_{row_count}_summary.json"

    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(OUTPUT_COLUMNS))
        writer.writeheader()
        writer.writerows(export.rows)

    displacement_values = [
        float(row[column])
        for row in export.rows
        for column in (*TRUE_F_COLUMNS, *PRED_F_COLUMNS)
    ]
    summary_record = {
        "column_count": len(OUTPUT_COLUMNS),
        "column_order": list(OUTPUT_COLUMNS),
        "dataset_path": str(dataset_csv),
        "direct_fem_substitution": True,
        "finite_values": True,
        "inference_only": True,
        "maximum_absolute_displacement_m": max(
            abs(value) for value in displacement_values
        ),
        "minima_order": "copied unchanged; canonical atan2 order from source",
        "model_path": str(model_path),
        "numeric_columns_only": True,
        "output_csv": str(output_csv),
        "predicted_fem_transform_verified": True,
        "row_count": row_count,
        "selection": f"first {row_count} source rows; no randomization",
        "shape_verified": f"{row_count}x{len(OUTPUT_COLUMNS)}",
        "transform_applied_to_true_and_predicted": (
            "F1,F2,F3,F4 = -[W3,W1,W4,W2]"
        ),
        "true_fem_transform_verified": True,
        "units": "metres",
        "wolfram_columns_in_output": False,
    }
    summary.write_text(
        json.dumps(
            summary_record, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    readme.write_text(
        "\n".join(
            (
                "# Direct FEM-order prediction dataset",
                "",
                "This is a direct FEM-order dataset generated by saved-model "
                "inference only. No new FEM solve, synthetic-data generation, "
                "calibration, or model training was run.",
                "",
                f"- Rows: {row_count} (the first {row_count} clean source rows; "
                "no randomization).",
                "- Columns: 22 numeric columns only.",
                "- All values are in metres.",
                "- Format: 8 true FEM displacements -> 6 equilibrium-minimum "
                "coordinates -> 8 predicted FEM displacements.",
                "- First 8 columns: true FEM displacements in F1, F2, F3, F4 order.",
                "- Middle 6 columns: the three equilibrium minima, copied unchanged "
                "in canonical atan2 order.",
                "- Last 8 columns: predicted FEM displacements in F1, F2, F3, F4 order.",
                "",
                "Both displacement blocks are ready for direct FEM substitution.",
                "Do not apply the Wolfram-to-FEM transform again.",
                "",
                "The already-applied transform is:",
                "",
                "`F1 = -W3, F2 = -W1, F3 = -W4, F4 = -W2`",
                "",
                f"Source dataset: `{dataset_csv.as_posix()}`",
                f"Saved model: `{model_path.as_posix()}`",
                "",
            )
        ),
        encoding="utf-8",
    )
    return output_csv, readme, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--n",
        "--rows",
        dest="n",
        type=int,
        default=DEFAULT_N,
        help="number of first clean source rows to export",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    export = build_export(arguments.dataset, arguments.model, arguments.n)
    validate_export(export, arguments.n)
    output_csv, readme, summary = write_outputs(
        export,
        dataset_csv=arguments.dataset,
        model_path=arguments.model,
        output_dir=arguments.output_dir,
    )
    print(f"output_csv={output_csv}")
    print(f"readme={readme}")
    print(f"summary={summary}")
    print(f"shape={len(export.rows)}x{len(OUTPUT_COLUMNS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
