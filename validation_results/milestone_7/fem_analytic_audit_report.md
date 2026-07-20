# Milestone 7: FEM numerical audit

## Scope

The audit preserves the all-positive four-electrode default physics.
It validates the shared P1 Laplace assembly on analytic problems, checks
Dirichlet markers and real-scale geometry, tests field sign and symmetry,
compares raw element fields with recovered-gradient minima, and reruns the
coherent E1-relative E2/E3-swapped reference benchmark under refinement.
Failed rows and flagged candidates remain in the CSV outputs.

## Analytic validation

| problem | quantity | value | tolerance | pass | samples | units |
|:---|:---|---:|---:|:---:|---:|:---|
| concentric-capacitor | potential-relative-l2 | 0.00016186404 | 0.02 | yes | 2175 | 1 |
| concentric-capacitor | potential-maximum-absolute-error | 0.00072742089 | 0.03 | yes | 2175 | V |
| concentric-capacitor | recovered-field-relative-l2 | 0.0037951314 | 0.1 | yes | 1936 | 1 |
| concentric-capacitor | recovered-field-magnitude-relative-l2 | 0.002730736 | 0.1 | yes | 1936 | 1 |
| concentric-capacitor | raw-element-field-relative-l2 | 0.028103046 | 0.1 | yes | 4120 | 1 |
| concentric-capacitor | minimum-outward-radial-field | 13.187556 | 0 | yes | 1936 | V/m |
| concentric-capacitor | relative-free-residual | 1.7891821e-15 | 1e-10 | yes | 2175 | 1 |
| concentric-capacitor | Dirichlet-boundary-error | 0 | 1e-12 | yes | 190 | V |
| uniform-field-disk | potential-maximum-absolute-error | 9.9920072e-16 | 1e-11 | yes | 545 | V |
| uniform-field-disk | field-maximum-vector-error | 7.7409548e-15 | 1e-10 | yes | 545 | V/m |
| uniform-field-disk | mean-electric-field-x | -1 | -1 | yes | 545 | V/m |
| symmetric-four-electrode | center-field-relative-to-probe-field | 0.00031496125 | 0.05 | yes | 1 | 1 |
| symmetric-four-electrode | potential-orbit-maximum-error | 6.9218174e-06 | 0.01 | yes | 12 | V |
| symmetric-four-electrode | field-rotation-relative-error | 0.016051501 | 0.1 | yes | 12 | 1 |
| symmetric-four-electrode | central-minimum-distance | 0.00028284271 | 0.001 | yes | 1 | m |

Analytic Laplace and sign tests pass: **yes**.
Undisplaced four-electrode symmetry tests at h=1 mm pass: **yes**.

## Boundary markers

| boundary | nodes | expected V | geometry residual (m) | potential error (V) | overlaps | missing | complete |
|:---|---:|---:|---:|---:|---:|---:|:---:|
| outer | 158 | 0 | 6.93889e-18 | 0 | 0 | 0 | yes |
| electrode-1 | 32 | 1 | 1.73472e-18 | 0 | 0 | 0 | yes |
| electrode-2 | 32 | 1 | 3.46945e-18 | 0 | 0 | 0 | yes |
| electrode-3 | 32 | 1 | 1.73472e-18 | 0 | 0 | 0 | yes |
| electrode-4 | 32 | 1 | 3.46945e-18 | 0 | 0 | 0 | yes |

All boundary markers are complete and exclusive: **yes**.

## Geometry

Gmsh constructs the vacuum as an exact OpenCASCADE outer disk minus
four exact circular disks; the P1 mesh represents each curve by chords.
The requested radius is 10 mm, the outer
radius is 50 mm, and the centre
radius is 21.48 mm.
Across nominal geometry and reference rows 1--10, the minimum electrode
gap is 9.60944 mm and minimum outer clearance is
17.6386 mm. All cases are valid: **yes**.

## Recovered-gradient candidate audit

Artifact action: `flag`. The default audit action only
flags; it never alters forward outputs. The documented flag requires a
candidate within 0.02 mesh lengths of a facet together with an adjacent
raw-field jump ratio of at least 0.50, or recovered psi at least 100 times
the lowest validated candidate in that row.
Flagged candidates: `13/39`.
Flagged selected minima: `4/30` (`13.333%`).
A flagged-selected fraction above 10% is conservatively treated as
numerically material for the scientific conclusion.
Raw P1 fields are constant per triangle and generally do not vanish at a
recovered-field zero; that disagreement is reported but is not alone an
artifact criterion.

## Reference-validation refinement

| h (mm) | rows | completed | exact-three | mean (mm) | median (mm) | max (mm) | p95 (mm) | runtime (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 10 | 10 | 5 | 1.08687 | 0.953208 | 5.98241 | 2.20448 | 22.222 |
| 1.5 | 10 | 10 | 9 | 1.08951 | 0.925194 | 5.92806 | 2.2958 | 26.128 |
| 1 | 10 | 10 | 9 | 1.07248 | 0.923011 | 5.90802 | 2.29576 | 32.301 |
| 0.75 | 10 | 10 | 8 | 1.08454 | 0.924076 | 5.92901 | 2.32722 | 37.469 |
| 0.5 | 3 | 3 | 3 | 0.922159 | 0.99991 | 1.29811 | 1.22659 | 18.663 |

Comparable rows 1--10 mean-error change from
h=2 mm to
h=0.75 mm is
`0.215%` reduction.
At least 5% improvement: **no**.
Exactly-three topology reaches `8/10` and is stable for every row:
**no**.
For the optional rows 1--3 h=0.5 mm check, mean error changes from
0.932388 mm at h=2 mm to 0.922159 mm, a `1.097%` reduction.

## Numerical-bug and scientific conclusion

No FEM assembly, sign, boundary-marker, or geometry bug was found.
It is not yet scientifically justified to call the remaining mismatch model-class limited under the documented numerical criteria.
Model-class conclusion justified: **no**.
No ML or synthetic dataset generation was performed.

Total audit runtime: `162.136 s`.
