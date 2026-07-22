"""Direct and CSV prediction interface for saved inverse RF-trap models."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .dataset import sort_points_by_polar_angle
from .export_prediction_dataset import load_prediction_model
from .inverse_training import INPUT_COLUMNS, MICROMETRES_PER_METRE, TARGET_COLUMNS


DEFAULT_MODEL_PATH = Path("validation_results/inverse_model_merged_51974/mlp.joblib")
CLOSED_LOOP_BEST_MODEL_PATH = Path(
    "validation_results/inverse_model_merged_29995/mlp.joblib"
)
TRAINING_DISPLACEMENT_LIMIT_M = 500.0e-6

PREDICTED_WOLFRAM_METRE_COLUMNS = tuple(
    f"pred_{column}" for column in TARGET_COLUMNS
)
PREDICTED_WOLFRAM_MICROMETRE_COLUMNS = tuple(
    f"pred_{column.removesuffix('_m')}_um" for column in TARGET_COLUMNS
)
FEM_METRE_COLUMNS = tuple(
    f"fem_f{electrode}_{component}_m"
    for electrode in range(1, 5)
    for component in ("dx", "dy")
)
FEM_MICROMETRE_COLUMNS = tuple(
    column.removesuffix("_m") + "_um" for column in FEM_METRE_COLUMNS
)
PREDICTION_OUTPUT_COLUMNS = (
    *INPUT_COLUMNS,
    *PREDICTED_WOLFRAM_METRE_COLUMNS,
    *PREDICTED_WOLFRAM_MICROMETRE_COLUMNS,
    *FEM_METRE_COLUMNS,
    *FEM_MICROMETRE_COLUMNS,
    "warning",
)

UNIT_SCALE_TO_METRES = {"m": 1.0, "mm": 1.0e-3}
WOLFRAM_ELECTRODE_LABELS = (
    "W1 upper-right",
    "W2 lower-right",
    "W3 upper-left",
    "W4 lower-left",
)
FEM_ELECTRODE_LABELS = (
    "F1 upper-left = -W3",
    "F2 upper-right = -W1",
    "F3 lower-left = -W4",
    "F4 lower-right = -W2",
)


@dataclass(frozen=True)
class InversePredictionBatch:
    """Model inputs and predicted displacements in both electrode orders."""

    minima_m: NDArray[np.float64]
    wolfram_displacements_m: NDArray[np.float64]
    fem_displacements_m: NDArray[np.float64]
    warnings: tuple[str, ...]
    auto_sort_enabled: bool
    sorting_applied: tuple[bool, ...]

    def __post_init__(self) -> None:
        count = self.minima_m.shape[0]
        expected = {
            "minima_m": (count, 6),
            "wolfram_displacements_m": (count, 8),
            "fem_displacements_m": (count, 8),
        }
        for name, shape in expected.items():
            array = np.asarray(getattr(self, name), dtype=float)
            if array.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain only finite values")
        if len(self.warnings) != count:
            raise ValueError("warnings must contain one value per prediction row")
        if len(self.sorting_applied) != count:
            raise ValueError("sorting_applied must contain one value per prediction row")


def convert_minima_to_metres(values: object, units: str) -> NDArray[np.float64]:
    """Convert finite minimum coordinates from metres or millimetres to metres."""

    if units not in UNIT_SCALE_TO_METRES:
        raise ValueError("units must be 'm' or 'mm'")
    array = np.asarray(values, dtype=float)
    if not array.size or not np.all(np.isfinite(array)):
        raise ValueError("minimum coordinates must be finite and non-empty")
    return np.asarray(array * UNIT_SCALE_TO_METRES[units], dtype=float)


def parse_minima_string(value: str, units: str = "m") -> NDArray[np.float64]:
    """Parse ``x1,y1;x2,y2;x3,y3`` and return shape ``(3, 2)`` in metres."""

    pairs = [item.strip() for item in value.split(";") if item.strip()]
    if len(pairs) != 3:
        raise ValueError("minima must contain exactly three semicolon-separated pairs")
    coordinates: list[list[float]] = []
    for pair in pairs:
        components = [item.strip() for item in pair.split(",")]
        if len(components) != 2:
            raise ValueError("each minimum must be an x,y pair")
        try:
            coordinates.append([float(components[0]), float(components[1])])
        except ValueError as error:
            raise ValueError("minimum coordinates must be numeric") from error
    return convert_minima_to_metres(coordinates, units).reshape(3, 2)


def canonical_order_minima(points: object) -> NDArray[np.float64]:
    """Return three minima in the training pipeline's canonical order.

    The forward and synthetic-data pipelines use :func:`sort_points_by_polar_angle`:
    increasing ``atan2(y, x)`` after wrapping angles into ``[0, 2*pi)``. Reusing
    that helper here keeps interactive predictions aligned with the training labels.
    """

    array = np.asarray(points, dtype=float)
    if array.shape != (3, 2):
        raise ValueError("points must have shape (3, 2)")
    if not np.all(np.isfinite(array)):
        raise ValueError("points must contain only finite values")
    return np.asarray(sort_points_by_polar_angle(array), dtype=float)


def wolfram_to_fem_displacements_m(values: object) -> NDArray[np.float64]:
    """Apply ``F1,F2,F3,F4 = -[W3,W1,W4,W2]`` and preserve input shape."""

    array = np.asarray(values, dtype=float)
    original_shape = array.shape
    if array.ndim >= 2 and array.shape[-2:] == (4, 2):
        electrode_pairs = array
    elif array.ndim >= 1 and array.shape[-1] == 8:
        electrode_pairs = array.reshape(*array.shape[:-1], 4, 2)
    else:
        raise ValueError("Wolfram displacements must end in shape (4,2) or (8,)")
    if not np.all(np.isfinite(electrode_pairs)):
        raise ValueError("Wolfram displacements must be finite")
    transformed = -electrode_pairs[..., [2, 0, 3, 1], :]
    return np.asarray(transformed.reshape(original_shape), dtype=float)


def load_minima_csv(path: str | Path, units: str = "m") -> NDArray[np.float64]:
    """Load the exact six-column minima schema and return ``(N, 6)`` metres."""

    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("input CSV is missing a header")
        missing = sorted(set(INPUT_COLUMNS).difference(reader.fieldnames))
        if missing:
            raise ValueError(f"input CSV is missing columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError("input CSV contains no rows")
    try:
        values = np.asarray(
            [[float(row[column]) for column in INPUT_COLUMNS] for row in rows],
            dtype=float,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("input CSV contains a malformed minimum coordinate") from error
    return convert_minima_to_metres(values, units).reshape(-1, 6)


def predict_inverse(
    model: object,
    minima_m: object,
    *,
    sort_minima: bool = True,
) -> InversePredictionBatch:
    """Predict displacements after optional canonical minima ordering."""

    minima = _minima_rows(minima_m)
    original_minima = minima.reshape(-1, 3, 2)
    if sort_minima:
        minima = np.asarray(
            [canonical_order_minima(row) for row in original_minima], dtype=float
        ).reshape(-1, 6)
    sorting_applied = tuple(
        not np.array_equal(before, after)
        for before, after in zip(
            original_minima, minima.reshape(-1, 3, 2), strict=True
        )
    )
    prediction = np.asarray(model.predict(minima), dtype=float)
    expected_shape = (minima.shape[0], 8)
    if prediction.shape != expected_shape:
        raise ValueError(
            f"model predictions must have shape {expected_shape}, got {prediction.shape}"
        )
    if not np.all(np.isfinite(prediction)):
        raise ValueError("model predictions contain NaN or infinite values")
    fem = wolfram_to_fem_displacements_m(prediction)
    warnings = tuple(_range_warning(row) for row in prediction)
    return InversePredictionBatch(
        minima,
        prediction,
        fem,
        warnings,
        bool(sort_minima),
        sorting_applied,
    )


def write_prediction_csv(
    prediction: InversePredictionBatch,
    path: str | Path,
) -> Path:
    """Write canonical metre/µm prediction columns and range warnings."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    wolfram_um = MICROMETRES_PER_METRE * prediction.wolfram_displacements_m
    fem_um = MICROMETRES_PER_METRE * prediction.fem_displacements_m
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(PREDICTION_OUTPUT_COLUMNS))
        writer.writeheader()
        for row_index in range(prediction.minima_m.shape[0]):
            row: dict[str, float | str] = {}
            row.update(zip(INPUT_COLUMNS, prediction.minima_m[row_index], strict=True))
            row.update(
                zip(
                    PREDICTED_WOLFRAM_METRE_COLUMNS,
                    prediction.wolfram_displacements_m[row_index],
                    strict=True,
                )
            )
            row.update(
                zip(
                    PREDICTED_WOLFRAM_MICROMETRE_COLUMNS,
                    wolfram_um[row_index],
                    strict=True,
                )
            )
            row.update(
                zip(FEM_METRE_COLUMNS, prediction.fem_displacements_m[row_index], strict=True)
            )
            row.update(zip(FEM_MICROMETRE_COLUMNS, fem_um[row_index], strict=True))
            row["warning"] = prediction.warnings[row_index]
            writer.writerow(row)
    return destination


def _minima_rows(values: object) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.shape == (3, 2) or array.shape == (6,):
        array = array.reshape(1, 6)
    elif array.ndim == 3 and array.shape[1:] == (3, 2):
        array = array.reshape(-1, 6)
    elif array.ndim != 2 or array.shape[1] != 6:
        raise ValueError("minima_m must have shape (3,2), (6,), (N,3,2), or (N,6)")
    if not np.all(np.isfinite(array)):
        raise ValueError("minima_m must contain only finite values")
    return np.asarray(array, dtype=float)


def _range_warning(wolfram_row_m: NDArray[np.float64]) -> str:
    outside = np.flatnonzero(np.abs(wolfram_row_m) > TRAINING_DISPLACEMENT_LIMIT_M)
    if not outside.size:
        return ""
    details = ", ".join(
        f"{TARGET_COLUMNS[index].removesuffix('_m').upper()}="
        f"{MICROMETRES_PER_METRE * wolfram_row_m[index]:.3f} um"
        for index in outside
    )
    return f"outside +/-500 um training range: {details}"


def format_prediction_text(prediction: InversePredictionBatch) -> str:
    """Format one or more predictions for terminal or GUI display."""

    sorting_label = "enabled" if prediction.auto_sort_enabled else "disabled"
    lines = [
        f"Prediction rows: {prediction.minima_m.shape[0]}",
        f"Auto-sort minima: {sorting_label}",
        "Canonical order: increasing polar angle atan2(y, x) wrapped to [0, 2*pi).",
    ]
    display_rows = min(prediction.minima_m.shape[0], 5)
    for row_index in range(display_rows):
        if prediction.minima_m.shape[0] > 1:
            lines.extend(("", f"Row {row_index + 1}:"))
        lines.append("Ordered minima used by model (m):")
        for minimum_index, pair in enumerate(
            prediction.minima_m[row_index].reshape(3, 2), start=1
        ):
            lines.append(
                f"  min{minimum_index}: x={pair[0]:.12g}, y={pair[1]:.12g}"
            )
        wolfram_um = (
            MICROMETRES_PER_METRE
            * prediction.wolfram_displacements_m[row_index].reshape(4, 2)
        )
        fem_um = (
            MICROMETRES_PER_METRE
            * prediction.fem_displacements_m[row_index].reshape(4, 2)
        )
        lines.append("Wolfram-order predicted displacements (um):")
        for label, pair in zip(WOLFRAM_ELECTRODE_LABELS, wolfram_um, strict=True):
            lines.append(f"  {label}: dx={pair[0]:.6f}, dy={pair[1]:.6f}")
        lines.append("FEM-order transformed displacements (um):")
        for label, pair in zip(FEM_ELECTRODE_LABELS, fem_um, strict=True):
            lines.append(f"  {label}: dx={pair[0]:.6f}, dy={pair[1]:.6f}")
        if prediction.warnings[row_index]:
            lines.append(f"WARNING: {prediction.warnings[row_index]}")
    if prediction.minima_m.shape[0] > display_rows:
        lines.append(f"... {prediction.minima_m.shape[0] - display_rows} more rows")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the direct/CSV saved inverse-model prediction CLI."""

    parser = argparse.ArgumentParser(
        prog="rf-trap-predict-inverse",
        description="Predict Wolfram-order electrode displacements from three minima.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--minima", help='three pairs formatted as "x1,y1;x2,y2;x3,y3"')
    inputs.add_argument("--input-csv", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--units", choices=tuple(UNIT_SCALE_TO_METRES), default="m")
    parser.add_argument(
        "--no-sort-minima",
        action="store_true",
        help="disable the pipeline's canonical polar-angle minima ordering",
    )
    return parser


def normalize_minima_cli_args(argv: Sequence[str] | None) -> list[str]:
    """Bind a negative-leading minima token to ``--minima`` for argparse."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    for index, value in enumerate(arguments[:-1]):
        if value == "--minima" and not arguments[index + 1].startswith("--"):
            arguments[index : index + 2] = [f"--minima={arguments[index + 1]}"]
            break
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    """Run model-only prediction for a direct string or canonical CSV input."""

    parser = build_parser()
    arguments = parser.parse_args(normalize_minima_cli_args(argv))
    if arguments.input_csv is not None and arguments.output_csv is None:
        parser.error("--output-csv is required with --input-csv")
    minima_m = (
        parse_minima_string(arguments.minima, arguments.units)
        if arguments.minima is not None
        else load_minima_csv(arguments.input_csv, arguments.units)
    )
    model = load_prediction_model(arguments.model)
    prediction = predict_inverse(model, minima_m, sort_minima=not arguments.no_sort_minima)
    print(format_prediction_text(prediction))
    if arguments.output_csv is not None:
        path = write_prediction_csv(prediction, arguments.output_csv)
        print(f"output_csv={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
