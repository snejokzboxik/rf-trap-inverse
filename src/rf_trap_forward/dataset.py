"""Ingestion and export of the Mathematica reference dataset."""

from __future__ import annotations

import argparse
import ast
import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

MinimaFrame = Literal["absolute", "electrode1-relative"]


@dataclass(frozen=True)
class ReferenceDatasetSummary:
    """Scalar format and range diagnostics for one parsed reference dataset."""

    row_count: int
    all_rows_have_four_displacement_pairs: bool
    all_rows_have_three_minima_pairs: bool
    mathematica_notation_row_count: int
    raw_displacement_min_m: float
    raw_displacement_max_m: float
    relative_displacement_min_m: float
    relative_displacement_max_m: float
    equilibrium_coordinate_min_m: float
    equilibrium_coordinate_max_m: float
    equilibrium_radius_min_m: float
    equilibrium_radius_median_m: float
    equilibrium_radius_max_m: float
    equilibrium_fraction_at_least_one_mm: float
    likely_raw_displacement_half_range_um: float
    closer_to_range_um: int


@dataclass(frozen=True)
class DatasetExportPaths:
    """Files written by :func:`export_reference_dataset`."""

    csv_path: Path
    npz_path: Path


@dataclass(frozen=True)
class ReferenceDataset:
    """Raw and derived coordinates from the quasi-equilibrium reference file.

    Raw displacements have shape ``(n, 4, 2)`` and raw absolute minima have
    shape ``(n, 3, 2)``. All stored numerical values use metres.
    """

    raw_displacements_m: NDArray[np.float64]
    raw_minima_absolute_m: NDArray[np.float64]
    source_path: Path | None = None
    mathematica_notation_rows: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Validate shapes and make independent finite floating-point arrays."""

        displacements = np.asarray(self.raw_displacements_m, dtype=float)
        minima = np.asarray(self.raw_minima_absolute_m, dtype=float)
        if displacements.ndim != 3 or displacements.shape[1:] != (4, 2):
            raise ValueError("raw_displacements_m must have shape (n, 4, 2)")
        if minima.ndim != 3 or minima.shape[1:] != (3, 2):
            raise ValueError("raw_minima_absolute_m must have shape (n, 3, 2)")
        if displacements.shape[0] != minima.shape[0] or displacements.shape[0] == 0:
            raise ValueError("displacement and minima arrays must have the same nonzero row count")
        if not np.all(np.isfinite(displacements)) or not np.all(np.isfinite(minima)):
            raise ValueError("reference dataset coordinates must be finite")
        if self.mathematica_notation_rows and any(
            row < 1 or row > displacements.shape[0]
            for row in self.mathematica_notation_rows
        ):
            raise ValueError("mathematica_notation_rows contains an invalid row number")
        object.__setattr__(self, "raw_displacements_m", displacements.copy())
        object.__setattr__(self, "raw_minima_absolute_m", minima.copy())
        if self.source_path is not None:
            object.__setattr__(self, "source_path", Path(self.source_path))

    @property
    def row_count(self) -> int:
        """Return the number of reference samples."""

        return int(self.raw_displacements_m.shape[0])

    @property
    def relative_displacements_m(self) -> NDArray[np.float64]:
        """Return electrode 2--4 displacements relative to electrode 1."""

        return displacements_relative_to_electrode1(self.raw_displacements_m)

    @property
    def relative_displacements_flat_m(self) -> NDArray[np.float64]:
        """Return relative displacement inputs in six-column forward-model order."""

        return self.relative_displacements_m.reshape(self.row_count, 6)

    @property
    def minima_relative_to_electrode1_m(self) -> NDArray[np.float64]:
        """Return each absolute equilibrium position translated by electrode 1."""

        return self.raw_minima_absolute_m - self.raw_displacements_m[:, :1, :]

    @property
    def minima_absolute_angle_sorted_m(self) -> NDArray[np.float64]:
        """Return absolute minima sorted by polar angle about the global origin."""

        return sort_points_by_polar_angle(self.raw_minima_absolute_m)

    @property
    def minima_relative_angle_sorted_m(self) -> NDArray[np.float64]:
        """Return electrode-1-relative minima sorted about electrode 1."""

        return sort_points_by_polar_angle(self.minima_relative_to_electrode1_m)

    def minima_m(
        self,
        *,
        relative_to_electrode1: bool = False,
        sort_by_polar_angle: bool = True,
    ) -> NDArray[np.float64]:
        """Return minima in the requested frame and optional angular ordering."""

        values = (
            self.minima_relative_to_electrode1_m
            if relative_to_electrode1
            else self.raw_minima_absolute_m
        )
        if sort_by_polar_angle:
            return sort_points_by_polar_angle(values)
        return values.copy()

    def summary(self) -> ReferenceDatasetSummary:
        """Compute format, range, and scale diagnostics in SI units."""

        displacement_limit_um = float(
            np.max(np.abs(self.raw_displacements_m)) * 1.0e6
        )
        relative = self.relative_displacements_m
        radii = np.linalg.norm(self.raw_minima_absolute_m, axis=2)
        closer_to = min((200, 500), key=lambda value: abs(displacement_limit_um - value))
        return ReferenceDatasetSummary(
            row_count=self.row_count,
            all_rows_have_four_displacement_pairs=(
                self.raw_displacements_m.shape[1:] == (4, 2)
            ),
            all_rows_have_three_minima_pairs=(
                self.raw_minima_absolute_m.shape[1:] == (3, 2)
            ),
            mathematica_notation_row_count=len(self.mathematica_notation_rows),
            raw_displacement_min_m=float(np.min(self.raw_displacements_m)),
            raw_displacement_max_m=float(np.max(self.raw_displacements_m)),
            relative_displacement_min_m=float(np.min(relative)),
            relative_displacement_max_m=float(np.max(relative)),
            equilibrium_coordinate_min_m=float(np.min(self.raw_minima_absolute_m)),
            equilibrium_coordinate_max_m=float(np.max(self.raw_minima_absolute_m)),
            equilibrium_radius_min_m=float(np.min(radii)),
            equilibrium_radius_median_m=float(np.median(radii)),
            equilibrium_radius_max_m=float(np.max(radii)),
            equilibrium_fraction_at_least_one_mm=float(np.mean(radii >= 1.0e-3)),
            likely_raw_displacement_half_range_um=displacement_limit_um,
            closer_to_range_um=closer_to,
        )


def parse_reference_row(line: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Parse one Mathematica-style mapping into ``(4, 2)`` and ``(3, 2)`` arrays."""

    if line.count("->") != 1:
        raise ValueError("reference row must contain exactly one '->' separator")
    displacement_text, minima_text = (part.strip() for part in line.split("->"))
    displacements = _parse_mathematica_array(displacement_text)
    minima = _parse_mathematica_array(minima_text)
    if displacements.shape != (4, 2):
        raise ValueError(
            f"reference row must contain four displacement pairs; got {displacements.shape}"
        )
    if minima.shape != (3, 2):
        raise ValueError(
            f"reference row must contain three equilibrium pairs; got {minima.shape}"
        )
    return displacements, minima


def load_reference_dataset(path: str | Path) -> ReferenceDataset:
    """Parse every nonblank row from a Mathematica reference-data text file."""

    source = Path(path)
    displacement_rows: list[NDArray[np.float64]] = []
    minima_rows: list[NDArray[np.float64]] = []
    mathematica_rows: list[int] = []
    logical_row = 0
    for physical_line, raw_line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        logical_row += 1
        try:
            displacements, minima = parse_reference_row(line)
        except ValueError as error:
            raise ValueError(f"invalid reference row at line {physical_line}: {error}") from error
        displacement_rows.append(displacements)
        minima_rows.append(minima)
        if "*^" in line:
            mathematica_rows.append(logical_row)
    if not displacement_rows:
        raise ValueError("reference dataset contains no rows")
    return ReferenceDataset(
        raw_displacements_m=np.stack(displacement_rows),
        raw_minima_absolute_m=np.stack(minima_rows),
        source_path=source,
        mathematica_notation_rows=tuple(mathematica_rows),
    )


def displacements_relative_to_electrode1(
    raw_displacements_m: ArrayLike,
) -> NDArray[np.float64]:
    """Convert raw 8D electrode displacements to three relative ``(x, y)`` pairs."""

    values = np.asarray(raw_displacements_m, dtype=float)
    if values.ndim < 2 or values.shape[-2:] != (4, 2):
        raise ValueError("raw_displacements_m must end with shape (4, 2)")
    if not np.all(np.isfinite(values)):
        raise ValueError("raw_displacements_m must be finite")
    return values[..., 1:, :] - values[..., :1, :]


def sort_points_by_polar_angle(points_m: ArrayLike) -> NDArray[np.float64]:
    """Sort the final point axis by angle in ``[0, 2*pi)`` about the origin."""

    points = np.asarray(points_m, dtype=float)
    if points.ndim < 2 or points.shape[-1] != 2:
        raise ValueError("points_m must end with a coordinate axis of length two")
    if not np.all(np.isfinite(points)):
        raise ValueError("points_m must be finite")
    angles = np.mod(np.arctan2(points[..., 1], points[..., 0]), 2.0 * np.pi)
    order = np.argsort(angles, axis=-1, kind="stable")
    return np.take_along_axis(points, order[..., np.newaxis], axis=-2)


def export_reference_dataset(
    dataset: ReferenceDataset,
    csv_path: str | Path,
    npz_path: str | Path,
    *,
    primary_minima_frame: MinimaFrame = "absolute",
) -> DatasetExportPaths:
    """Export raw, relative, and angle-sorted coordinates to CSV and compressed NPZ."""

    if primary_minima_frame not in ("absolute", "electrode1-relative"):
        raise ValueError("primary_minima_frame must be 'absolute' or 'electrode1-relative'")
    csv_output = Path(csv_path)
    npz_output = Path(npz_path)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    npz_output.parent.mkdir(parents=True, exist_ok=True)
    primary = dataset.minima_m(
        relative_to_electrode1=(primary_minima_frame == "electrode1-relative"),
        sort_by_polar_angle=True,
    )
    rows = _dataset_csv_rows(dataset, primary, primary_minima_frame)
    with csv_output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        npz_output,
        raw_displacements_m=dataset.raw_displacements_m,
        relative_displacements_m=dataset.relative_displacements_m,
        relative_displacements_flat_m=dataset.relative_displacements_flat_m,
        raw_minima_absolute_m=dataset.raw_minima_absolute_m,
        minima_absolute_angle_sorted_m=dataset.minima_absolute_angle_sorted_m,
        minima_relative_to_electrode1_m=dataset.minima_relative_to_electrode1_m,
        minima_relative_angle_sorted_m=dataset.minima_relative_angle_sorted_m,
        primary_minima_m=primary,
        primary_minima_frame=np.asarray(primary_minima_frame),
    )
    return DatasetExportPaths(csv_path=csv_output, npz_path=npz_output)


def write_dataset_summary(
    dataset: ReferenceDataset,
    path: str | Path,
) -> Path:
    """Write a Markdown format-and-range verification report."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_dataset_summary_markdown(dataset), encoding="utf-8")
    return output


def _parse_mathematica_array(text: str) -> NDArray[np.float64]:
    translated = text.replace("*^", "e").replace("{", "[").replace("}", "]")
    try:
        value = ast.literal_eval(translated)
        array = np.asarray(value, dtype=float)
    except (SyntaxError, ValueError, TypeError) as error:
        raise ValueError("invalid Mathematica numeric-list syntax") from error
    if not np.all(np.isfinite(array)):
        raise ValueError("reference row contains a non-finite number")
    return array


def _dataset_csv_rows(
    dataset: ReferenceDataset,
    primary_minima_m: NDArray[np.float64],
    primary_minima_frame: MinimaFrame,
) -> list[dict[str, object]]:
    relative = dataset.relative_displacements_m
    absolute_sorted = dataset.minima_absolute_angle_sorted_m
    relative_minima = dataset.minima_relative_to_electrode1_m
    relative_sorted = dataset.minima_relative_angle_sorted_m
    rows: list[dict[str, object]] = []
    for sample in range(dataset.row_count):
        row: dict[str, object] = {
            "sample_index": sample + 1,
            "primary_minima_frame": primary_minima_frame,
        }
        _add_pairs(row, "raw_d", dataset.raw_displacements_m[sample], start_index=1)
        _add_pairs(row, "relative_d", relative[sample], start_index=2)
        _add_pairs(
            row,
            "raw_minimum_absolute_",
            dataset.raw_minima_absolute_m[sample],
            start_index=1,
        )
        _add_pairs(
            row,
            "minimum_absolute_sorted_",
            absolute_sorted[sample],
            start_index=1,
        )
        _add_pairs(
            row,
            "minimum_relative_unsorted_",
            relative_minima[sample],
            start_index=1,
        )
        _add_pairs(
            row,
            "minimum_relative_sorted_",
            relative_sorted[sample],
            start_index=1,
        )
        _add_pairs(row, "minimum_", primary_minima_m[sample], start_index=1)
        rows.append(row)
    return rows


def _add_pairs(
    row: dict[str, object],
    prefix: str,
    pairs_m: NDArray[np.float64],
    *,
    start_index: int,
) -> None:
    for offset, pair in enumerate(pairs_m):
        index = start_index + offset
        row[f"{prefix}{index}_x_m"] = pair[0]
        row[f"{prefix}{index}_y_m"] = pair[1]


def _dataset_summary_markdown(dataset: ReferenceDataset) -> str:
    summary = dataset.summary()
    first_rows = min(dataset.row_count, 10)
    lines = [
        "# Reference dataset verification",
        "",
        "## Structural checks",
        "",
        f"- Parsed rows: `{summary.row_count}`",
        f"- Every row has four displacement pairs: **{_yes_no(summary.all_rows_have_four_displacement_pairs)}**",
        f"- Every row has three equilibrium-position pairs: **{_yes_no(summary.all_rows_have_three_minima_pairs)}**",
        f"- Rows using Mathematica `*^` exponents: `{summary.mathematica_notation_row_count}`",
        "",
        "The first ten nonblank rows were individually parsed and shape-checked:",
        "",
        "| row | displacement pairs | equilibrium pairs | Mathematica exponent | status |",
        "|---:|---:|---:|:---:|:---:|",
    ]
    notation_rows = set(dataset.mathematica_notation_rows)
    for row in range(1, first_rows + 1):
        lines.append(
            f"| {row} | 4 | 3 | {_yes_no(row in notation_rows)} | valid |"
        )
    lines.extend(
        [
            "",
            "### First ten parsed rows",
            "",
            "The following normalized values are in metres:",
            "",
            "```text",
        ]
    )
    for row in range(first_rows):
        lines.append(
            f"{row + 1}: {_format_pairs(dataset.raw_displacements_m[row])} -> "
            f"{_format_pairs(dataset.raw_minima_absolute_m[row])}"
        )
    lines.extend(
        [
            "```",
            "",
            "## Ranges and inferred units",
            "",
            "The source text contains no unit declaration. Per the supplied project context,",
            "values are interpreted as metres; their magnitudes are consistent with that",
            "interpretation.",
            "",
            f"- Raw displacement coordinate range: `{summary.raw_displacement_min_m * 1.0e6:.9g}` to `{summary.raw_displacement_max_m * 1.0e6:.9g} µm`.",
            f"- Electrode-1-relative displacement range: `{summary.relative_displacement_min_m * 1.0e6:.9g}` to `{summary.relative_displacement_max_m * 1.0e6:.9g} µm`.",
            f"- Observed raw half-range: approximately `±{summary.likely_raw_displacement_half_range_um:.6g} µm`, much closer to `±{summary.closer_to_range_um} µm` than `±{500 if summary.closer_to_range_um == 200 else 200} µm`.",
            f"- Absolute equilibrium-coordinate range: `{summary.equilibrium_coordinate_min_m * 1.0e3:.9g}` to `{summary.equilibrium_coordinate_max_m * 1.0e3:.9g} mm`.",
            f"- Equilibrium radial-distance range: `{summary.equilibrium_radius_min_m * 1.0e3:.9g}` to `{summary.equilibrium_radius_max_m * 1.0e3:.9g} mm`; median `{summary.equilibrium_radius_median_m * 1.0e3:.9g} mm`.",
            f"- Fraction of equilibrium positions at least 1 mm from the origin: `{summary.equilibrium_fraction_at_least_one_mm:.3%}`.",
            "",
            "## Coordinate convention",
            "",
            "The file is ingested as an 8D raw frame; electrode 1 is not assumed fixed.",
            "Forward-model inputs are derived as `(d2-d1, d3-d1, d4-d1)`. Absolute",
            "and electrode-1-relative minima are both retained and independently sorted",
            "by polar angle in `[0, 2π)` in their respective frames.",
            "",
        ]
    )
    return "\n".join(lines)


def _format_pairs(pairs: NDArray[np.float64]) -> str:
    return "{" + ",".join(
        f"{{{float(pair[0]):.17g},{float(pair[1]):.17g}}}" for pair in pairs
    ) + "}"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def build_parser() -> argparse.ArgumentParser:
    """Build the reference-dataset verification command-line parser."""

    parser = argparse.ArgumentParser(
        prog="rf-trap-reference-dataset",
        description="Parse, verify, and export the Mathematica reference dataset.",
    )
    parser.add_argument("input", nargs="?", type=Path, default=Path("Data.txt"))
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("validation_results") / "milestone_3",
    )
    parser.add_argument(
        "--primary-minima-frame",
        choices=("absolute", "electrode1-relative"),
        default="absolute",
        help="frame used by the generic minimum_1..3 CSV columns; both frames are always exported",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Verify the supplied reference file and write CSV, NPZ, and Markdown outputs."""

    arguments = build_parser().parse_args(argv)
    dataset = load_reference_dataset(arguments.input)
    output = arguments.output_directory
    paths = export_reference_dataset(
        dataset,
        output / "reference_dataset.csv",
        output / "reference_dataset.npz",
        primary_minima_frame=arguments.primary_minima_frame,
    )
    report_path = write_dataset_summary(dataset, output / "dataset_verification.md")
    summary = dataset.summary()
    print(f"rows: {summary.row_count}")
    print(
        "raw displacement range (um): "
        f"{summary.raw_displacement_min_m * 1.0e6:.9g}, "
        f"{summary.raw_displacement_max_m * 1.0e6:.9g}"
    )
    print(
        "equilibrium coordinate range (mm): "
        f"{summary.equilibrium_coordinate_min_m * 1.0e3:.9g}, "
        f"{summary.equilibrium_coordinate_max_m * 1.0e3:.9g}"
    )
    print(f"CSV: {paths.csv_path}")
    print(f"NPZ: {paths.npz_path}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
