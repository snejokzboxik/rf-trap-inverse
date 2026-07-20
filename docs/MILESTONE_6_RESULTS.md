# Milestone 6 results: model-hypothesis diagnosis

## Outcome

Milestone 6 does not validate the current four-electrode FEM model against the
reference dataset. The best rows 1--50 diagnostic completes all 50 rows, but
only 36 rows have exactly three pre-selection Hessian-valid minima. Its mean
error is 1.27046 mm and its maximum error is 5.92593 mm. The conservative gate
requires exactly-three topology for every row, mean error no greater than
0.25 mm, and maximum error no greater than 0.5 mm.

Large synthetic dataset generation therefore remains unsafe. No ML, inverse
model, or synthetic dataset generation was implemented.

The earlier focused conclusion is unchanged: the fourth candidate in the old
60 micrometre, 4.0 mm-boundary demonstrator case is a recovered-gradient
interpolation artifact. Milestone 6 concerns a different, real-scale model and
does not reinterpret that result.

## Study design

The production screen used reference rows 1--10, a 2.0 mm characteristic mesh,
the named real-scale geometry, a fixed 50 mm outer radius, and a +/-8 mm search
square. FEM rows ran in fresh interpreter processes. The best three distinct
output hypotheses were then evaluated on rows 1--50 when runtime remained
reasonable.

Because electrode 1 remains the source reference, the numbering screen covered
all six permutations of source electrodes E2--E4 while keeping source E1 in FEM
slot E1. Both absolute and electrode-1-relative displacement inputs were tested.
Reference minima were interpreted both absolutely and after subtracting the E1
displacement. Eight global transforms covered identity, x and y sign flips,
x/y exchange, rotations by 90, 180, and 270 degrees, and reflection across the
anti-diagonal.

For each usable transform/frame combination, an optional positive global output
scale was fitted on rows 1--10. That scale was held fixed during promotion. An
explicit inner-radius normalization and conversion back to metres was also
tested; its maximum round-trip error was 4.34e-19 m, confirming that it is only
a change of representation.

The model screen included all-positive electrodes, all seven nontrivial binary
polarity patterns modulo a global sign, and a fitted linear combination of four
one-electrode basis fields. Alternate cases are diagnostics only: the package's
default physical model was not changed.

The saved tables contain 612 screening output hypotheses and three promoted
hypotheses, 6,270 per-row records, and 16,092 matched-minimum records. Failures
were retained rather than omitted.

## Scale evidence

For reference rows 1--10, absolute minimum radii span 0.460977--5.12352 mm with
a median of 3.50509 mm. The median is 0.305322 times the 11.48 mm inner radius
and 0.163179 times the 21.48 mm electrode-centre radius. In the E1-relative
frame the radii span 0.670894--5.34136 mm with a median of 3.88548 mm.

The best promoted output scale is 0.925496632, a 7.45% radial correction. It
improves the rows 1--50 mean only from 1.28175 to 1.27046 mm within the fitted
basis family. A simple scale mismatch therefore cannot explain the residual.

## Numbering, frame, and orientation evidence

The best numbering map remains the Milestone 5 diagnostic:

`FEM E1,E2,E3,E4 <- source E1,E3,E2,E4`.

The identity global coordinate transform ranks best. Sign flips, x/y exchange,
and rigid rotations do not reveal a missing orientation convention. The best
screening score uses relative displacement inputs but absolute reference
outputs. This mixed-frame interpretation changes the mean only slightly and is
not a coherent physical convention; it is retained transparently as diagnostic
evidence rather than adopted as a default.

## Polarity and basis-field evidence

All seven nontrivial binary-polarity variants are worse than the all-positive
case or fail to return three minima. In particular, the alternating checkerboard
pattern returns no completed rows in the screen.

The one-electrode basis-field fit succeeds on all ten screening rows and returns
the normalized voltage vector

`(0.999926798, 0.999809418, 0.999786521, 1.0)`.

Its components differ by at most 0.02135%, so the optimization effectively
converges back to all-positive electrodes. The small Gram eigenvalue ratio,
5.14520e-8, establishes a near-null field combination at the supplied points,
but the ordinary forward minima pipeline still has large spatial outliers and
unstable candidate count. It is not evidence for a new polarity model.

## Best metrics

The best rows 1--10 screen is the fitted-basis, E2/E3-swapped, relative-input,
absolute-reference, identity-transform, unscaled diagnostic:

| metric | result |
|:---|---:|
| completed rows | 10/10 |
| exactly-three rows | 6/10 |
| mean error | 1.03867 mm |
| maximum error | 5.90503 mm |

On the same ten rows, that mean is 4.435% below the Milestone 5 mean of
1.08687 mm. The maximum and topology failures remain decisive.

The best promoted rows 1--50 hypothesis uses the same fitted basis and
coordinate interpretation plus the fixed 0.925496632 output scale:

| metric | result |
|:---|---:|
| completed rows | 50/50 |
| exactly-three rows | 36/50 |
| mean error | 1.27046 mm |
| median error | 1.05612 mm |
| maximum error | 5.92593 mm |
| p95 error | 2.45948 mm |
| validation gate | fail |

The promoted all-positive comparison has mean error 1.28335 mm, so fitted
voltage and scale diagnostics improve it by only 1.005% on the same 50 rows.
The 50-row mean is not directly comparable with Milestone 5's ten-row mean.

## Diagnosis and decision

Electrode numbering matters, but orientation does not. Output scale and
absolute-versus-relative frame choices provide only small corrections. Binary
polarity alternatives are worse, and the continuous basis fit collapses to the
all-positive vector. The remaining mismatch is therefore primarily a model
class/topology problem: the real-scale four-electrode Laplace model and its
recovered-gradient minima pipeline do not reproduce all three reference
branches across the selected rows.

The best case fails both spatial thresholds and the exactly-three topology
requirement. The current model is not safe for synthetic dataset generation.

## Artifacts

The complete evidence is in `validation_results/milestone_6`:

- `hypothesis_summary.csv`: aggregate metrics and gate decision;
- `hypothesis_rows.csv`: completion, topology, residual, mesh, and row errors;
- `hypothesis_minima.csv`: matched coordinates, error vectors, radial errors;
- `basis_fit.csv`: one-hot field-fit diagnostics;
- `scale_diagnostics.csv`: physical-scale and normalization checks;
- `milestone_6_report.md`: focused report and hypothesis tables;
- `plots/`: reference/computed overlays, per-row error distributions, error
  vectors, and radial comparisons for all three promoted hypotheses.

The production study runtime was 530.803 seconds on the development machine.
