# Milestone 5 results: real-scale validation and mismatch diagnosis

Date: 2026-07-20

## Configuration

The reference benchmark now defaults to the supplied real-scale geometry:

- outer-boundary radius: 50 mm;
- electrode radius: 10 mm;
- centre-to-nearest-surface inner radius: 11.48 mm;
- electrode-centre radius: 21.48 mm;
- diagonal coordinate: `a = 21.48 mm / sqrt(2) = 15.1886537 mm`;
- E1 `(-a,+a)`, E2 `(+a,+a)`, E3 `(-a,-a)`, E4 `(+a,-a)`;
- minima-search square: ±8 mm in both coordinates;
- outer boundary: 0 V.

The retained physical-model default is +1 V on every electrode. Checkerboard
`(+1,-1,-1,+1) V` was tested only as a named diagnostic. The old small-scale
demonstrator remains available through `--geometry demonstrator` and in its
regression tests.

Every production solve ran in a fresh interpreter. All distances were calculated
in metres and are displayed below in millimetres. Computed and reference minima
were paired by minimum-total-distance assignment.

## Mesh refinement with identity numbering

Rows 1–10 all completed at each requested real-scale mesh size:

| mesh size | rows completed | exactly-three topology | mean error | median error | max error | p95 error |
|---:|---:|---:|---:|---:|---:|---:|
| 2.0 mm | 10/10 | 5/10 | 1.47368 mm | 1.18772 mm | 6.20446 mm | 2.44766 mm |
| 1.5 mm | 10/10 | 8/10 | 1.45172 mm | 1.19340 mm | 6.19067 mm | 2.38598 mm |
| 1.0 mm | 10/10 | 9/10 | 1.44011 mm | 1.14295 mm | 6.19812 mm | 2.33100 mm |

Mean error changes by only 0.03357 mm from 2.0 to 1.0 mm, while the maximum
remains about 6.2 mm. The strict three-minimum structure is therefore not stable
over all rows and refinements, although it improves to 9/10 rows at 1.0 mm. Mesh
refinement alone does not resolve the dataset mismatch.

## Convention and polarity diagnostics

At 2.0 mm mesh, the identity absolute-frame case gives 1.49647 mm mean error,
slightly worse than the 1.47368 mm E1-relative result. This indicates that the
coordinate-frame choice is not the main remaining mismatch.

All five non-identity permutations of source electrodes E2–E4 were screened on
rows 1–3. Exchanging source E2 and E3 was best and was promoted to rows 1–10:

- FEM E1, E2, E3, E4 receive source E1, E3, E2, E4 displacements;
- E1-relative input and output coordinates;
- all electrodes at +1 V;
- 2.0 mm mesh;
- 10/10 completed rows;
- 1.08687 mm mean, 0.953208 mm median, 5.98241 mm maximum, and 2.20448 mm p95
  matched error;
- exactly three pre-selection Hessian-valid candidates in only 5/10 rows.

This is the best tested convention, but the permutation remains a diagnostic
hypothesis rather than a proven dataset numbering map.

Both checkerboard alternating-polarity variants failed all ten rows because the
four-electrode model produced one validated minimum rather than the required
three. This is a topology result, not a suppressed validation failure. It is
consistent with the project warning that the reference article's eight-rod
octupole must not be assumed equivalent to this four-electrode FEM boundary-value
problem.

## Comparison with Milestone 4

The best mean error falls from 3.18585 mm to 1.08687 mm, a 65.884% reduction, and
all ten rows now complete in the best case. The corrected scale, 50 mm boundary,
and ±8 mm search region remove the demonstrated domain/search exclusions from
Milestone 4.

The remaining errors are nevertheless large: the median is about 0.95 mm, the
maximum is about 5.98 mm, and half the best-case rows have extra Hessian-valid
candidates before the API selects the lowest three. The current model is closer
to the reference data, but is not quantitatively or topologically consistent.

## Decision

Large synthetic dataset generation is **not safe yet**. The report uses an
explicit conservative gate: all ten rows must complete with exactly three
physical minima, mean error at most 0.25 mm, and maximum error at most 0.5 mm.
The gate is a reporting decision criterion, not a fitted physical parameter.

No ML, inverse model, or synthetic dataset generation was implemented. The core
all-positive physical model was retained. Per-electrode potentials, coordinate
frames, and numbering permutations are exposed only as explicit diagnostics.

The earlier fourth-candidate conclusion is unchanged: the extra candidate in the
60 µm, 4.0 mm demonstrator case remains classified as a recovered-gradient
interpolation artifact.

## Artifacts

`validation_results/milestone_5` contains:

- `variant_summary.csv`;
- `validation_rows.csv`;
- `validation_minima.csv`;
- `milestone_5_report.md`;
- `best_case/reference_validation_rows.csv`;
- `best_case/reference_validation_minima.csv`;
- `best_case/reference_validation_report.md`;
- `best_case/plots/row_0001.png` through `row_0010.png`.

The complete diagnostic run took 158.350 seconds on the milestone workstation.
