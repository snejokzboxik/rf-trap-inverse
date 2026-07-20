# Milestone 8 results: robust minima post-processing

## Decision

The selected-minimum interpolation flags found in Milestone 7 do not explain a
significant fraction of the reference-dataset mismatch. Robust selection fixes
the tested candidate-count instability, but the best-mapping mean error changes
from 1.08687 mm to 1.08754 mm, a 0.061% worsening. The maximum remains about
5.99 mm, far above the 0.5 mm validation threshold.

The remaining mismatch can now be attributed to physical model class/topology
with substantially stronger numerical justification. This is not a claim of
exact mesh invariance: three coarse 2.0-to-1.5 mm branch shifts exceed 0.25 mm.
All finer transitions pass, and the robust selected set is exactly three and
free of the documented robust artifact criteria at every tested mesh.

The conservative validation gate still fails. Synthetic dataset generation
remains unsafe. No ML, inverse model, new physical model, or synthetic dataset
was implemented, and the recovered-gradient default remains unchanged.

## Explicit modes

The post-processing API now has three named modes:

1. `recovered-gradient` is the unchanged legacy coarse scan, L-BFGS-B
   refinement, merge, single-stencil Hessian validation, and lowest-three
   selection.
2. `raw-element-diagnostic` reports local lows of the exact elementwise-constant
   P1 electric field. Its element-centroid representatives are diagnostics, not
   claimed sub-element field zeros.
3. `robust` combines multiple candidate sources and applies the documented
   quality tests before selecting three outputs.

The forward API still defaults to `recovered-gradient`. Robust and raw modes
must be requested explicitly.

## Candidate sources and preserved diagnostics

Robust mode starts from:

- all unique candidates from the legacy recovered-field search, including
  candidates that fail its original Hessian test;
- exact zeros of the continuous recovered P1 vector field inside individual
  triangles, computed in barycentric coordinates;
- local lows of raw element `|E|²`, refined against the recovered field.

Coincident sources are merged with their names and support counts retained.
Every merged candidate is written to `candidate_quality_table.csv`, including
rejected candidates. The table contains coordinates, nearest internal-facet and
electrode distances, recovered `|E|²`, containing and adjacent raw-element field
magnitudes, raw-field jump, four Hessian stencils and eigenvalue pairs, stability
class, legacy and robust flags, artifact score, acceptance, selection, and the
full classification reason.

The reported `artifact_probability` is a transparent rule-based score, not a
calibrated statistical probability. It adds 0.25 for close-facet location, 0.20
for a large adjacent raw-field jump, 0.40 for Hessian instability, and 0.15 for
high recovered `|E|²`, capped at 1. Labels are low below 0.30, medium below 0.60,
and high otherwise. Acceptance is determined by the explicit criteria below,
not by fitting this score to `Data.txt`.

## Robust criteria

Hessians are evaluated at 0.005, 0.0125, 0.025, and 0.05 times the actual mesh
parameter. At least three stencils must be valid; all finite eigenvalues must be
positive; and the largest per-eigenvalue variation ratio must not exceed 8.

A candidate is rejected as facet-sensitive only when all three conditions hold:

- internal-facet distance is at most 0.02 mesh parameters;
- adjacent raw-field jump is at least 0.50 of the larger adjacent magnitude;
- the multi-stencil Hessian is unstable.

Recovered `|E|²` must also be no more than 100 times the low-candidate scale.
The scale is the median of the lowest configured candidate set with a documented
floating-point floor. The expected three-output topology is therefore used only
to define a relative low-field scale; rejected candidates remain visible and
agreement with `Data.txt` is never part of acceptance.

## Rows 1--10 at h=2 mm

| mode | mapping | completed | exactly three | mean (mm) | median (mm) | max (mm) | p95 (mm) | selected flags | rejected |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| old recovered | identity | 10/10 | 5/10 | 1.47368 | 1.18772 | 6.20446 | 2.44766 | 0 | 0 |
| audit flag | identity | 10/10 | 5/10 | 1.47368 | 1.18772 | 6.20446 | 2.44766 | 5 | 0 |
| robust | identity | 10/10 | 10/10 | 1.47427 | 1.18580 | 6.20776 | 2.44758 | 0 | 26 |
| old recovered | E1,E3,E2,E4 | 10/10 | 5/10 | 1.08687 | 0.953208 | 5.98241 | 2.20448 | 0 | 0 |
| audit flag | E1,E3,E2,E4 | 10/10 | 5/10 | 1.08687 | 0.953208 | 5.98241 | 2.20448 | 4 | 0 |
| robust | E1,E3,E2,E4 | 10/10 | 10/10 | 1.08754 | 0.956273 | 5.98525 | 2.20475 | 0 | 24 |

Audit mode reproduces the four Milestone 7 best-mapping flags on rows 3, 4, 5,
and 10 without changing their coordinates. Robust mode retains those legacy
flags in its table but does not classify their exact cell-zero counterparts as
artifacts: their Hessians remain positive and stable across the four stencils.

Under the best mapping, robust selection rejects 24 high-field or unstable
candidates and accepts exactly three per row. Its 0.061% mean-error change is
well below the documented 5% materiality threshold. The flags therefore affected
topology reporting but were not a meaningful cause of the spatial error.

## Mesh consistency

| h (mm) | rows | completed | exactly three | mean (mm) | max (mm) | selected robust flags | rejected |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.00 | 10 | 10 | 10 | 1.08754 | 5.98525 | 0 | 24 |
| 1.50 | 10 | 10 | 10 | 1.09009 | 5.92913 | 0 | 10 |
| 1.00 | 10 | 10 | 10 | 1.07300 | 5.90851 | 0 | 7 |
| 0.75 | 10 | 10 | 10 | 1.08506 | 5.92962 | 0 | 8 |
| 0.50 | 3 | 3 | 3 | 0.921771 | 1.29588 | 0 | 0 |

Minimum-distance assignment tracks 99 branches between successive meshes. Of
those, 96 remain within 0.25 mm. The three exceptions are the first 2.0-to-1.5 mm
transition: two branches in row 6 and one in row 8, with a maximum shift of
0.299800 mm. All 69 transitions whose coarse mesh is 1.5 mm or finer pass; their
maximum shift is 0.164062 mm.

Robust topology is therefore stable across every tested row and mesh, while
branch coordinates become stable at the documented tolerance after the coarsest
mesh transition.

## Scientific interpretation

The robust method improves topology from 5/10 to 10/10 without improving
reference agreement. This cleanly separates two issues:

- recovered-gradient post-processing created extra high-field or unstable
  candidates and made the old exactly-three diagnostic unreliable;
- those candidates were not responsible for the approximately 1.09 mm mean
  spatial mismatch or the approximately 5.99 mm maximum outlier.

Together with the Milestone 7 analytic FEM validation and the unchanged error
plateau, the evidence now supports a model-class/topology limitation much more
strongly. The qualification about the three coarse branch shifts remains, but it
does not rescue the failed spatial validation.

The validation gate requires all rows complete, exactly-three topology on every
row, mean error no greater than 0.25 mm, and maximum error no greater than
0.5 mm. Robust selection satisfies completion and topology but fails both error
thresholds. Large synthetic dataset generation is not scientifically justified.

## Artifacts

All outputs are under `validation_results/milestone_8`:

- `milestone_8_report.md`;
- `robust_minima_summary.csv`;
- `candidate_quality_table.csv`;
- `old_vs_robust_reference_validation.csv`;
- `mesh_branch_stability.csv`;
- eight plots under `plots/`.

The production study used fresh-process FEM execution and took 300.510 seconds
on the development machine. All requested failures, rejected candidates, and
legacy flags are retained.
