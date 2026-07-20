# Reference dataset format

## Source grammar

`Data.txt` contains one Mathematica-style rule per non-empty line:

```text
{{dx1,dy1},{dx2,dy2},{dx3,dy3},{dx4,dy4}} -> {{x1,y1},{x2,y2},{x3,y3}}
```

The left side contains exactly four electrode-displacement pairs. The right side
contains exactly three quasi-equilibrium, or pseudopotential-minimum, position
pairs. The parser accepts ordinary decimal/scientific notation and Mathematica's
scientific notation, for example `5.449003089273541*^-6`.

The file itself has no units header. Following the supplied dataset description,
the parser treats every coordinate as metres. The magnitudes are consistent with
that interpretation: electrode displacements are approximately ±500 µm and most
equilibrium positions are several millimetres from the origin.

## Verified dimensions and ranges

The supplied file has 326 rows. Every row has four input pairs and three output
pairs, producing arrays with these shapes:

- raw displacements: `(326, 4, 2)`;
- raw absolute equilibrium positions: `(326, 3, 2)`.

Across all displacement components, the range is
`[-499.443487, 499.507690] µm`. This is clearly closer to ±500 µm than ±200 µm.
Across all equilibrium-position coordinates, the range is
`[-5.760000, 6.480000] mm`; 99.489% of equilibrium positions have radial distance
at least 1 mm from the origin.

These statistics describe the supplied file, not a generated or inferred
dataset. The complete verification, including the first ten rows, is recorded in
`validation_results/milestone_3/dataset_verification.md`.

## Coordinate conventions

Raw displacements are retained exactly as parsed. The six translation-invariant
input coordinates use electrode 1 as their reference:

```text
relative_displacements = [d2 - d1, d3 - d1, d4 - d1]
```

Equivalently, for `i` in `{2, 3, 4}`:

```text
rel_dxi = dxi - dx1
rel_dyi = dyi - dy1
```

The resulting array has shape `(n, 3, 2)` and its flattened model-input view has
shape `(n, 6)`. The original eight displacement coordinates remain available;
conversion never overwrites them.

The source equilibrium positions are not assumed to be in the electrode-1
reference frame. Both forms are stored:

```text
minima_absolute = source minima
minima_relative = source minima - displacement_1
```

For either frame, the three points can be sorted by increasing polar angle after
mapping `atan2(y, x)` to the interval `[0, 2π)`. This is the convention used by the
forward solver. Sorting is performed within each row and does not alter the raw
source ordering.

## Exports

Run:

```powershell
rf-trap-reference-dataset Data.txt --output-directory validation_results/milestone_3
```

The command writes:

- `reference_dataset.csv`: one row per configuration, including raw and relative
  displacements, raw and angle-sorted minima in both frames, and the selected
  primary minima frame;
- `reference_dataset.npz`: named NumPy arrays for the same representations;
- `dataset_verification.md`: source-shape, notation, range, and first-ten-row
  verification.

Use `--primary-minima-frame electrode1-relative` when downstream consumers should
use electrode-1-relative minima as their primary output. Absolute and relative
arrays are always exported regardless of this option.

This milestone validates and converts the reference data only. It does not imply
that the current four-electrode FEM model reproduces these positions.
