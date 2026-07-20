# Milestone 3 results: reference dataset integration

Date: 2026-07-20

## Outcome

The supplied `Data.txt` reference dataset is now parsed, validated, transformed,
and exported without changing the FEM or meshing model. The source contains 326
configurations; every row has exactly four electrode-displacement pairs and three
equilibrium-position pairs.

The parser supports Mathematica scientific notation such as `*^-6`. It retains
raw displacements with shape `(326, 4, 2)` and raw absolute minima with shape
`(326, 3, 2)`. It also provides six displacement coordinates relative to
electrode 1, both absolute and electrode-1-relative minima, and per-row
polar-angle sorting consistent with the forward solver.

## Dataset verification

| Quantity | Result |
|---|---:|
| Rows | 326 |
| Rows with Mathematica `*^` notation | 57 |
| All rows contain 4 input pairs | Yes |
| All rows contain 3 output pairs | Yes |
| Raw displacement range | -499.443487 to 499.507690 µm |
| Equilibrium-coordinate range | -5.760000 to 6.480000 mm |
| Equilibrium radial-distance range | 0.145602 to 6.480278 mm |
| Equilibrium radial-distance median | 3.811948 mm |
| Equilibria at least 1 mm from origin | 99.489% |
| Displacement scale | Closer to ±500 µm than ±200 µm |

The source file has no units header. All values are interpreted as metres per the
supplied dataset description, and the measured magnitudes are consistent with
that interpretation. No assumption is made that the source minima are already in
the electrode-1 reference frame.

## Outputs

`validation_results/milestone_3` contains:

- `reference_dataset.csv`;
- `reference_dataset.npz`;
- `dataset_verification.md`.

The CSV and NPZ include raw data as well as derived coordinate representations.
The Markdown report includes the parsed first ten source rows and full-file
range/shape checks.

## Tests and scope

The full test suite passes: 24 tests passed, 0 failed. Six Milestone 3 tests cover
a complete source row, Mathematica exponent notation, the exact source row count,
the 8D-to-6D transformation, polar-angle sorting, and CSV/NPZ export.

The prior focused candidate investigation remains unchanged: the fourth
Hessian-valid candidate in the 60 µm, 4.0 mm case is classified as a
recovered-gradient interpolation artifact. No physical-model, FEM, or meshing
change was made. No ML, inverse modeling, synthetic data generation, or bulk
dataset generation was implemented.
