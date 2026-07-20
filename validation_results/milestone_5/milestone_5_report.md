# Milestone 5: real-scale FEM/reference validation

## Geometry and numerical setup

- Outer-boundary radius: `50 mm`
- Electrode radius: `10 mm`
- Inner radius (centre to nearest surface): `11.48 mm`
- Electrode-centre radius: `21.48 mm`
- Diagonal coordinate `a`: `15.1886537 mm`
- Numbering: `E1=(-a,+a), E2=(+a,+a), E3=(-a,-a), E4=(+a,-a)`.
- Search square: `±8 mm` in x and y.
- Full-row coarse mesh sizes: `2 mm, 1.5 mm, 1 mm`.
- The alternating diagnostic uses checkerboard potentials
  `(+1, -1, -1, +1) V` in E1--E4 order. It is not a default change.

## Case summary

| case | scope | rows ok | exact-three rows | mean (mm) | median (mm) | max (mm) | p95 (mm) | runtime (s) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| relative_all_positive_identity_h2mm | full | 10/10 | 5 | 1.47368 | 1.18772 | 6.20446 | 2.44766 | 19.282 |
| relative_all_positive_identity_h1.5mm | full | 10/10 | 8 | 1.45172 | 1.1934 | 6.19067 | 2.38598 | 19.253 |
| relative_all_positive_identity_h1mm | full | 10/10 | 9 | 1.44011 | 1.14295 | 6.19812 | 2.331 | 22.549 |
| absolute_all_positive_identity_h2mm | full | 10/10 | 6 | 1.49647 | 1.18895 | 6.06861 | 2.29854 | 17.424 |
| electrode1-relative_alternating_identity_h2mm | full | 0/10 | 0 | n/a | n/a | n/a | n/a | 17.462 |
| absolute_alternating_identity_h2mm | full | 0/10 | 0 | n/a | n/a | n/a | n/a | 16.854 |
| relative_all_positive_perm_1243_h2mm | screen | 3/3 | 2 | 2.33233 | 2.43291 | 5.33817 | 4.50217 | 5.750 |
| relative_all_positive_perm_1324_h2mm | screen | 3/3 | 2 | 0.932388 | 0.9764 | 1.47949 | 1.38953 | 5.378 |
| relative_all_positive_perm_1342_h2mm | screen | 3/3 | 3 | 3.23288 | 3.22789 | 5.29344 | 5.06156 | 5.775 |
| relative_all_positive_perm_1423_h2mm | screen | 3/3 | 2 | 2.08853 | 2.55536 | 3.32284 | 3.23221 | 5.214 |
| relative_all_positive_perm_1432_h2mm | screen | 3/3 | 3 | 3.16354 | 3.25592 | 4.81485 | 4.64735 | 5.101 |
| relative_all_positive_perm_1324_h2mm | full-promoted | 10/10 | 5 | 1.08687 | 0.953208 | 5.98241 | 2.20448 | 18.303 |

## Diagnosis

For identity numbering, refinement from 2.0 to 1.0 mm changes mean error from
1.47368 to 1.44011 mm. Exactly-three topology improves from 5/10 to 9/10, so it
is not stable across all rows/refinements and refinement alone does not remove
the mismatch.

The best comparable full-row case is `relative_all_positive_perm_1324_h2mm` with 10/10 completed rows, mean error `1.08687 mm`, median `0.953208 mm`, maximum `5.98241 mm`, and p95 `2.20448 mm`.
Its FEM meshes contain 2168--2185 nodes and 4056--4090 triangles; the maximum
relative free residual is `1.18233e-15`.

Against the Milestone-4 mean of `3.18585 mm`, this is a `65.884%` reduction. The full CSV tables retain failed rows; no failed validation is hidden.

The absolute and electrode-1-relative cases explicitly test coordinate
origin handling. Permutation cases change only the source-to-FEM mapping
for E2--E4. The alternating-polarity cases test a clean diagnostic
Dirichlet variant without changing the all-positive default.
The best map is FEM E1--E4 <- source E1,E3,E2,E4; it is a diagnostic hypothesis,
not a proven dataset convention.

Alternating-polarity failures reported validated-minimum counts [1]; this topology does not supply the required three minima in the tested four-electrode model.

The reference article concerns an eight-rod octupole. Real-scale
dimensions can improve coordinate scale agreement, but do not establish
physical equivalence of the present four-electrode boundary-value model.

## Dataset-generation decision

**NOT SAFE YET:** the conservative validation gate is not met.
The gate requires all ten rows to complete with exactly three physical
minima, mean error at most 0.25 mm, and maximum error at most 0.5 mm.
This gate is an explicit project decision criterion, not a fitted model
parameter. No ML or synthetic dataset generation was performed.

Detailed CSVs and the best-case per-row plots are stored beside this
report under `best_case/`.

Total diagnostic runtime: `158.350 s`.
