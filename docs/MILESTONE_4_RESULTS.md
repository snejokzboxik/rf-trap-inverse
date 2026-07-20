# Milestone 4 results: FEM-to-reference validation

Date: 2026-07-20

## Benchmark definition

The default benchmark evaluated source rows 1–10 from `Data.txt`. Each solve used
the unchanged provisional demonstrator configuration:

- nominal electrode-centre radius: 1.1 mm;
- electrode radius: 0.32 mm;
- mesh characteristic length: 80 µm;
- circular outer-boundary radius: 4.0 mm;
- minima-search half-extent: 0.7 mm;
- all four electrodes at equal normalized potential and the outer boundary at
  zero potential.

Each production row ran in a fresh interpreter to prevent Gmsh history from
coupling configurations. The FEM fixes electrode 1, so the raw source inputs were
converted to `(d2-d1, d3-d1, d4-d1)`. Reference minima were translated by the
same origin change, `minimum_absolute-d1`. Computed and reference minima were
paired using minimum-total-distance assignment, not polar-order indices.

## Results

| Metric | Result |
|---|---:|
| Selected rows | 10 |
| Completed rows with 3 returned minima | 8 |
| Failed/incomplete rows | 2 |
| Rows with exactly 3 pre-selection Hessian-valid candidates | 7 |
| Matched minima | 24 |
| Mean matched error | 3,185.85 µm / 3.18585 mm |
| Median matched error | 3,089.38 µm / 3.08938 mm |
| Maximum matched error | 4,934.35 µm / 4.93435 mm |
| 95th-percentile matched error | 4,559.83 µm / 4.55983 mm |

Row 1 was rejected because the provisional electrode disks overlap after applying
that row's relative displacements. Row 3 found only two Hessian-valid minima and
raised the existing explicit `MinimaSearchError`. Row 9 returned three selected
minima but had five pre-selection Hessian-valid candidates, so it does not meet
the strict exactly-three topology diagnostic. No candidate classification or
selection rule was changed.

## Scale and boundary evidence

- 29 of 30 reference minima lie outside the configured ±0.7 mm search square.
- 13 of 30 reference minima lie outside the 4.0 mm FEM outer circle.
- Reference radial distances span 0.670894–5.34136 mm, with a 3.88548 mm median.
- Computed radial distances span 0.145615–0.847820 mm.
- The largest electrode-1 translation applied to the selected rows is only
  0.543111 mm, far below the approximately 3–5 mm matched errors.

These facts explain why minimum-distance assignment cannot produce quantitative
agreement: most reference solutions are outside the region in which the current
demonstrator searches, and many are outside its modeled vacuum domain entirely.

## Likely causes and convention checks

- **Absolute versus electrode-1-relative coordinates:** handled explicitly and
  therefore not the main residual mismatch. The maximum applied translation is
  too small to explain the errors.
- **Coordinate origin:** tied consistently to electrode 1 for both inputs and
  outputs. A remaining origin ambiguity cannot by itself explain the observed
  scale separation.
- **Electrode numbering:** the source ordering is assumed to match the FEM's
  cardinal ordering, but the supplied material does not establish that mapping.
  It remains an unresolved validation input.
- **Geometry scale:** a primary likely cause. The current radius and nominal
  centres are provisional and are incompatible with most reference positions.
- **Outer boundary and search region:** primary demonstrated causes. The current
  numerical domain excludes or cannot search most reference points.
- **Polarity/physical model:** a primary unresolved cause. The current FEM has
  four equal-phase electrodes; the supplied article discusses an eight-rod
  alternating-polarity octupole. Their equivalence must not be assumed.

## Conclusion

The current provisional four-electrode FEM model is **not consistent with the
reference dataset at the tested conventions and scales**. It is **not safe to
proceed to large synthetic dataset generation**. Before generation, the project
needs the physical electrode radius and nominal centres, the dataset's electrode
numbering and polarity convention, and outer-boundary/search dimensions that
contain the reference minima. The benchmark should then be rerun before any
inverse-model or ML work.

This conclusion does not alter the earlier focused result: the fourth candidate
in the 60 µm, 4.0 mm milestone-2 case remains classified as a recovered-gradient
interpolation artifact. No physical equation, FEM discretization, meshing rule,
or minimum-selection rule was changed in milestone 4.

## Artifacts and tests

`validation_results/milestone_4` contains:

- `reference_validation_rows.csv`;
- `reference_validation_minima.csv`;
- `reference_validation_report.md`;
- `plots/row_0001.png` through `plots/row_0010.png`.

The full test suite passes: 30 tests passed, 0 failed. Six milestone-4 tests use
mocked forward results to cover default/range/random row selection, coordinate
conversion, spatial assignment, summary statistics, failure retention, CSV and
Markdown serialization, and headless plot generation.
