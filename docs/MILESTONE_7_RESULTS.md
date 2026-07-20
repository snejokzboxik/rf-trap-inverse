# Milestone 7 results: FEM numerical audit

## Decision

The P1 Laplace FEM assembly passes the analytic capacitor and exact-linear
tests. The electric-field sign is correct, all Dirichlet boundary nodes are
classified exactly once, imposed voltages are exact at marked nodes, and the
real-scale geometry matches the requested circular-hole construction and
dimensions. No assembly, sign, boundary-condition, or geometry bug was found,
and no default physics was changed.

Nevertheless, it is **not yet scientifically justified** to say that the full
remaining mismatch is model-class limited. Reference error does not materially
decrease under refinement, which supports the model-class hypothesis, but the
recovered-gradient audit flags 4 of the 30 selected h=2 mm minima under the
documented conservative criterion. Exactly-three topology is also non-monotone
with mesh size. The potential solve is validated; the recovered-field/minima
post-processing remains a numerically material confounder.

Synthetic dataset generation therefore remains unsafe. No ML, inverse model,
new physical model, or synthetic dataset was implemented.

## Audit-preserving refactors

The production trap solver and analytic tests now share one public generic P1
Dirichlet-Laplace assembly. The production result at h=2 mm is unchanged from
Milestone 5 (`1.086874 mm` mean error), so this refactor is not a hidden model
change.

The exact per-triangle P1 electric field and nearest-facet calculation are also
shared by the forward-field recovery, the old extra-candidate investigation,
and the new artifact audit. Geometry clearance reporting is now a public,
testable calculation used by the existing overlap/containment rejection logic.

## Analytic validation

### Concentric circular capacitor

The audit solves an annulus with inner radius 10 mm at 1 V and outer radius
50 mm at 0 V on a 2 mm characteristic mesh. The comparison uses

`phi(r) = log(R/r) / log(R/a)`

and the outward field

`E(x,y) = (x,y) / (r^2 log(R/a))`.

| metric | result | audit tolerance |
|:---|---:|---:|
| potential relative L2 error | 1.61864e-4 | 2.0e-2 |
| potential maximum error | 7.27421e-4 V | 3.0e-2 V |
| recovered-field relative L2 error | 3.79513e-3 | 1.0e-1 |
| recovered `|E|` relative L2 error | 2.73074e-3 | 1.0e-1 |
| raw element-field relative L2 error | 2.81030e-2 | 1.0e-1 |
| minimum audited outward radial field | 13.1876 V/m | greater than 0 |
| relative free-node residual | 1.78918e-15 | 1.0e-10 |
| Dirichlet boundary error | 0 V | 1.0e-12 V |

The recovered field has the correct outward direction at every audited annulus
node. Nodes within 1.5 mesh lengths of either circular boundary are excluded
from the recovered-field norm because nodal averaging there is one-sided; this
approximation is explicit in the CSV.

### Uniform-field disk and sign

The boundary potential `phi=x` is harmonic and exactly representable by P1
elements. The expected field is `E=(-1,0)`.

| metric | result |
|:---|---:|
| maximum potential error | 9.99201e-16 V |
| maximum field-vector error | 7.74095e-15 V/m |
| mean Ex | -1.0 V/m |

This test would fail if the code used `+grad(phi)`. The implemented convention
is correctly `E=-grad(phi)`.

## Symmetric four-electrode audit

The undisplaced real-scale model was audited at h=1 mm. The finer symmetry mesh
is intentional: at h=2 mm an unstructured mesh produces a 12.7% recovered-field
rotation error even though potential symmetry is already good; that error
decreases to 1.61% at h=1 mm.

| metric | h=1 mm result | tolerance |
|:---|---:|---:|
| central field / probe-field scale | 3.14961e-4 | 5.0e-2 |
| maximum potential orbit error | 6.92182e-6 V | 1.0e-2 V |
| field rotation-equivariance error | 1.60515e-2 | 1.0e-1 |
| central minimum distance | 0.282843 mm | 1.0 mm |
| Hessian-valid candidates | 1 | diagnostic |

The potential and field converge toward the expected D4 symmetry, and the
one-minimum diagnostic finds the degenerate symmetric zero near the centre.
The default three-minimum expectation was not changed.

## Boundary-condition audit

At h=2 mm the real-scale mesh has 158 outer-boundary nodes and 32 nodes on each
electrode boundary. The union contains every mesh-boundary node exactly once:

| boundary | nodes | expected voltage | maximum radial residual | voltage error |
|:---|---:|---:|---:|---:|
| outer | 158 | 0 V | 6.93889e-18 m | 0 V |
| E1 | 32 | 1 V | 1.73472e-18 m | 0 V |
| E2 | 32 | 1 V | 3.46945e-18 m | 0 V |
| E3 | 32 | 1 V | 1.73472e-18 m | 0 V |
| E4 | 32 | 1 V | 3.46945e-18 m | 0 V |

Missing marker nodes: 0. Overlapping marker assignments: 0. The outer boundary
and electrode voltages are therefore assigned correctly.

## Geometry audit

Gmsh uses an OpenCASCADE Boolean difference: one exact 50 mm disk minus four
exact 10 mm circular disks. First-order triangles approximate the circular
curves by straight chords; this remains the documented geometric discretization.

The nominal electrode-centre radius is `11.48 + 10 = 21.48 mm`, with centres
on the requested diagonals. Nominal geometry and the E2/E3-mapped relative
displacements for reference rows 1--10 all pass explicit overlap and containment
checks. Across those 11 cases:

- minimum electrode-to-electrode surface gap: 9.60944 mm;
- minimum electrode-to-outer-boundary gap: 17.6386 mm;
- maximum centre-construction error: 0 m.

No electrode overlaps another or leaves the outer disk.

## Reference validation under refinement

The coherent Milestone 5 convention was retained: all-positive electrodes,
electrode-1-relative inputs and outputs, and
`FEM E1,E2,E3,E4 <- source E1,E3,E2,E4`. Every production row ran in a fresh
interpreter process.

| h (mm) | rows | completed | exactly three | mean (mm) | median (mm) | max (mm) | p95 (mm) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.00 | 10 | 10 | 5 | 1.08687 | 0.953208 | 5.98241 | 2.20448 |
| 1.50 | 10 | 10 | 9 | 1.08951 | 0.925194 | 5.92806 | 2.29580 |
| 1.00 | 10 | 10 | 9 | 1.07248 | 0.923011 | 5.90802 | 2.29576 |
| 0.75 | 10 | 10 | 8 | 1.08454 | 0.924076 | 5.92901 | 2.32722 |
| 0.50 | 3 | 3 | 3 | 0.922159 | 0.999910 | 1.29811 | 1.22659 |

From 2.0 to 0.75 mm, the comparable rows 1--10 mean improves by only 0.215%,
far below the documented 5% threshold. The best full-row mean occurs at 1.0 mm
but is only 1.325% below the 2.0 mm result. The trend is non-monotone.

For the optional rows 1--3 comparison, the h=2 mm mean is 0.932388 mm and the
h=0.5 mm mean is 0.922159 mm, only a 1.097% reduction. Topology changes
`5/10 -> 9/10 -> 9/10 -> 8/10`; it improves relative to h=2 mm but does not
stabilize monotonically or reach 10/10.

Mesh refinement therefore does not move the FEM solution meaningfully toward
the reference dataset.

## Raw-element versus recovered-gradient minima

The audit records all 39 pre-selection Hessian-valid candidates at h=2 mm, not
only the selected 30. For each it stores recovered `|E|^2`, the containing
triangle's raw P1 field, nearest-facet distance, adjacent raw-field jump, forward
selection status, artifact reasons, and retained/filtered status.

The default action is `flag`; it never changes the forward API. A candidate is
flagged when either:

1. it lies within 0.02 actual mesh parameters of a facet and the adjacent raw
   fields differ by at least 0.50 of the larger local raw-field magnitude; or
2. its recovered `|E|^2` is at least 100 times the lowest validated candidate
   in the row.

CLI action `filter` marks flagged candidates as not retained in the audit table,
but still does not silently change the production result.

Results:

- 13/39 Hessian-valid candidates are flagged;
- 4/30 selected minima are flagged (13.333%);
- no artifact-audit row failed;
- the four selected flags occur in rows 3, 4, 5, and 10 and satisfy the
  facet-lock plus raw-field-jump criterion.

Raw element fields normally do not vanish inside a P1 triangle, so a large raw
field at a recovered-field zero is reported but is not sufficient by itself to
classify an artifact. The combined facet and jump evidence is the conservative
criterion. Because more than 10% of selected minima are flagged, recovered-field
post-processing remains numerically material.

## Scientific conclusion

The potential FEM, sign convention, boundary conditions, and geometry are
numerically correct at the tested tolerances. There was no FEM bug to fix.
Reference errors also plateau under refinement, providing evidence for a
physical model-class mismatch.

However, the selected-minimum artifact rate and non-monotone topology prevent a
clean attribution of the entire mismatch to the model class. The defensible
conclusion after Milestone 7 is:

> The four-electrode physical model is likely incomplete relative to the
> reference dataset, but recovered-gradient/minima post-processing must be
> made more robust before the residual can be called scientifically
> model-class limited without qualification.

## Artifacts

All evidence is stored under `validation_results/milestone_7`:

- `fem_analytic_audit_report.md`;
- `analytic_error_summary.csv`;
- `boundary_marker_summary.csv`;
- `geometry_sanity_summary.csv`;
- `mesh_refinement_reference_validation.csv`;
- `symmetry_audit_summary.csv`;
- `minima_interpolation_diagnostics.csv`;
- six plots under `plots/`.

The production audit runtime was 162.136 seconds. All requested failed-row and
artifact records are retained.
