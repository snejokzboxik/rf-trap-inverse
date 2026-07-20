# Milestone 6: model-hypothesis diagnosis

## Scope and invariant model assumptions

The screen uses rows 1--10, a 2.0 mm characteristic mesh, the named
50 mm real-scale outer boundary, 10 mm electrodes, 11.48 mm inner
surface radius, and a +/-8 mm minima-search square. Production FEM rows
run in fresh interpreter processes. Failed rows remain in every table.

The default physical model remains four all-positive electrodes and a
grounded outer boundary. Every alternate voltage vector, coordinate
transform, numbering map, and fitted scale in this report is diagnostic
only. No default was changed.

Electrode 1 remains the source reference. Consequently, all six
permutations of source E2--E4 were tested while source E1 stayed mapped
to FEM E1. The eight output transforms cover identity, independent x/y
sign flips, x/y swap, 90/180/270 degree rotations, and anti-diagonal
reflection.

## Scale diagnostics

| reference frame | radius min / median / max (mm) | median / inner radius | median / electrode-centre radius | inner-radius round-trip error (m) |
|:---|:---|---:|---:|---:|
| absolute | 0.460977 / 3.50509 / 5.12352 | 0.305322 | 0.163179 | 4.34e-19 |
| electrode1-relative | 0.670894 / 3.88548 / 5.34136 | 0.338457 | 0.180888 | 2.17e-19 |

Dividing reference positions by the inner radius and multiplying by
the same radius is exactly a change of representation: the measured
round-trip error above is numerical zero and cannot improve agreement.
The separate fitted-output-scale hypotheses multiply FEM predictions
by one positive scalar fitted on rows 1--10; promoted rows retain that
fixed scalar rather than refitting on rows 11--50.

## One-electrode basis-field fit

- Successful basis rows: `10/10`
- Fitted E1--E4 diagnostic potentials: `(0.999926798, 0.999809418, 0.999786521, 1) V`
- Gram eigenvalues: `(0.00845454326, 21816.0905, 69523.7519, 72979.3471)`
- Smallest eigenvalue / Gram trace: `5.14519506e-08`
- Basis-fit runtime: `15.922 s`

Each one-hot basis solves Laplace's equation on the same row mesh.
Linearity then fits one global voltage vector that minimizes field
magnitude at the transformed reference minima. The fitted vector is
subsequently tested through the ordinary forward minima pipeline; a
small field-fit eigenvalue alone is not treated as validation.

The fitted vector differs from all-positive by at most 0.02135%. This is
evidence that the basis fit converged back to the all-positive model, not
evidence for a materially different polarity convention.

## Polarity/model screen at the best numbering map

The table reports the best output interpretation for each voltage vector at
the E2/E3-swapped, electrode-1-relative input map. The complete transform and
failure records remain in the CSV files.

| potentials E1--E4 | model | rows ok | exact-three | best mean (mm) | best max (mm) |
|:---|:---|---:|---:|---:|---:|
| ++++ | all-positive | 10/10 | 5 | 1.05679 | 5.95449 |
| +++- | binary polarity | 8/10 | 6 | 3.32299 | 5.55945 |
| ++-+ | binary polarity | 7/10 | 3 | 3.28875 | 5.11918 |
| ++-- | binary polarity | 0/10 | 0 | n/a | n/a |
| +-++ | binary polarity | 5/10 | 4 | 3.40050 | 5.38595 |
| +-+- | binary polarity | 0/10 | 0 | n/a | n/a |
| +--+ | alternating checkerboard | 0/10 | 0 | n/a | n/a |
| +--- | binary polarity | 8/10 | 3 | 3.31551 | 5.59865 |
| fitted `(0.999927,0.999809,0.999787,1)` | basis linear combination | 10/10 | 6 | 1.03867 | 5.90503 |

All seven nontrivial binary patterns are worse than all-positive or fail to
return three minima. The minute fitted-voltage deviations improve the screened
mean by only 0.0181 mm and do not fix the outlier or topology failures.

## Promoted hypotheses (rows 1--50)

| rank | hypothesis | rows ok | exact-three | mean (mm) | median (mm) | max (mm) | p95 (mm) | gate |
|---:|:---|---:|---:|---:|---:|---:|---:|:---:|
| 1 | `basis_fitted_relative_perm1324_pppp__ref-absolute__identity__scale-fitted` | 50/50 | 36 | 1.27046 | 1.05612 | 5.92593 | 2.45948 | no |
| 2 | `basis_fitted_relative_perm1324_pppp__ref-absolute__identity__scale-none` | 50/50 | 36 | 1.28175 | 1.13008 | 6.10309 | 2.53037 | no |
| 3 | `relative_perm1324_pppp__ref-absolute__identity__scale-none` | 50/50 | 30 | 1.28335 | 1.1257 | 6.26397 | 2.49896 | no |

## Screening hypothesis table (top 50 of full CSV)

The complete screening table is `hypothesis_summary.csv`. It includes
every tested model/frame/transform/scale hypothesis and all failed
cases. The top 50 are reproduced here for a readable Markdown audit.

| rank | family | input | map | reference | transform | scale mode / value | rows ok | exact-three | mean (mm) | max (mm) |
|---:|:---|:---|:---|:---|:---|:---|---:|---:|---:|---:|
| 1 | basis-fitted-linear-combination | electrode1-relative | 1-3-2-4 | absolute | identity | none / 1 | 10/10 | 6 | 1.03867 | 5.90503 |
| 2 | basis-fitted-linear-combination+output-scale-fit | electrode1-relative | 1-3-2-4 | absolute | identity | fitted / 0.92549663 | 10/10 | 6 | 1.053 | 5.69596 |
| 3 | all-positive | electrode1-relative | 1-3-2-4 | absolute | identity | none / 1 | 10/10 | 5 | 1.05679 | 5.95449 |
| 4 | all-positive+output-scale-fit | electrode1-relative | 1-3-2-4 | absolute | identity | fitted / 0.9159028 | 10/10 | 5 | 1.07388 | 5.71163 |
| 5 | basis-fitted-linear-combination+inner-radius-roundtrip | electrode1-relative | 1-3-2-4 | electrode1-relative | identity | inner-radius-roundtrip / 1 | 10/10 | 6 | 1.08052 | 5.93315 |
| 6 | basis-fitted-linear-combination | electrode1-relative | 1-3-2-4 | electrode1-relative | identity | none / 1 | 10/10 | 6 | 1.08052 | 5.93315 |
| 7 | basis-fitted-linear-combination+output-scale-fit | electrode1-relative | 1-3-2-4 | electrode1-relative | identity | fitted / 0.93242439 | 10/10 | 6 | 1.08674 | 5.7176 |
| 8 | all-positive+inner-radius-roundtrip | electrode1-relative | 1-3-2-4 | electrode1-relative | identity | inner-radius-roundtrip / 1 | 10/10 | 5 | 1.08687 | 5.98241 |
| 9 | all-positive | electrode1-relative | 1-3-2-4 | electrode1-relative | identity | none / 1 | 10/10 | 5 | 1.08687 | 5.98241 |
| 10 | all-positive+output-scale-fit | electrode1-relative | 1-3-2-4 | electrode1-relative | identity | fitted / 0.92329856 | 10/10 | 5 | 1.1094 | 5.73408 |
| 11 | all-positive+inner-radius-roundtrip | absolute | 1-3-2-4 | absolute | identity | inner-radius-roundtrip / 1 | 10/10 | 5 | 1.11316 | 5.61249 |
| 12 | all-positive | absolute | 1-3-2-4 | absolute | identity | none / 1 | 10/10 | 5 | 1.11316 | 5.61249 |
| 13 | all-positive+output-scale-fit | absolute | 1-3-2-4 | absolute | identity | fitted / 0.92544909 | 10/10 | 5 | 1.11843 | 5.39823 |
| 14 | all-positive+output-scale-fit | absolute | 1-3-2-4 | electrode1-relative | identity | fitted / 0.91716501 | 10/10 | 5 | 1.29204 | 5.41065 |
| 15 | all-positive | absolute | 1-3-2-4 | electrode1-relative | identity | none / 1 | 10/10 | 5 | 1.298 | 5.65503 |
| 16 | all-positive+output-scale-fit | electrode1-relative | 1-2-3-4 | absolute | identity | fitted / 0.87189039 | 10/10 | 5 | 1.45308 | 5.67798 |
| 17 | all-positive+output-scale-fit | electrode1-relative | 1-2-3-4 | electrode1-relative | identity | fitted / 0.87849875 | 10/10 | 5 | 1.4709 | 5.76721 |
| 18 | all-positive+inner-radius-roundtrip | electrode1-relative | 1-2-3-4 | electrode1-relative | identity | inner-radius-roundtrip / 1 | 10/10 | 5 | 1.47368 | 6.20446 |
| 19 | all-positive | electrode1-relative | 1-2-3-4 | electrode1-relative | identity | none / 1 | 10/10 | 5 | 1.47368 | 6.20446 |
| 20 | all-positive | electrode1-relative | 1-2-3-4 | absolute | identity | none / 1 | 10/10 | 5 | 1.47819 | 6.12677 |
| 21 | all-positive+output-scale-fit | absolute | 1-2-3-4 | absolute | identity | fitted / 0.87472765 | 10/10 | 6 | 1.4819 | 5.61884 |
| 22 | all-positive+inner-radius-roundtrip | absolute | 1-2-3-4 | absolute | identity | inner-radius-roundtrip / 1 | 10/10 | 6 | 1.49647 | 6.06861 |
| 23 | all-positive | absolute | 1-2-3-4 | absolute | identity | none / 1 | 10/10 | 6 | 1.49647 | 6.06861 |
| 24 | all-positive+output-scale-fit | absolute | 1-2-3-4 | electrode1-relative | identity | fitted / 0.86833582 | 10/10 | 6 | 1.57276 | 5.69055 |
| 25 | all-positive | absolute | 1-2-3-4 | electrode1-relative | identity | none / 1 | 10/10 | 6 | 1.5745 | 6.1771 |
| 26 | all-positive | absolute | 1-4-2-3 | absolute | rotate-180 | none / 1 | 10/10 | 8 | 1.69858 | 5.81076 |
| 27 | all-positive | absolute | 1-4-2-3 | electrode1-relative | rotate-180 | none / 1 | 10/10 | 8 | 1.76809 | 5.85399 |
| 28 | all-positive+output-scale-fit | absolute | 1-4-2-3 | absolute | rotate-180 | fitted / 0.81121984 | 10/10 | 8 | 1.7702 | 5.23754 |
| 29 | all-positive+output-scale-fit | electrode1-relative | 1-2-4-3 | absolute | swap-xy | fitted / 0.80504465 | 10/10 | 8 | 1.78636 | 4.86271 |
| 30 | all-positive+output-scale-fit | absolute | 1-4-2-3 | electrode1-relative | rotate-180 | fitted / 0.81844938 | 10/10 | 8 | 1.79881 | 5.28682 |
| 31 | all-positive+output-scale-fit | absolute | 1-2-4-3 | absolute | swap-xy | fitted / 0.81458975 | 10/10 | 7 | 1.80291 | 5.0361 |
| 32 | all-positive | electrode1-relative | 1-4-2-3 | absolute | rotate-180 | none / 1 | 10/10 | 9 | 1.81797 | 5.63589 |
| 33 | all-positive | absolute | 1-2-4-3 | absolute | swap-xy | none / 1 | 10/10 | 7 | 1.82501 | 5.62557 |
| 34 | basis-fitted-linear-combination+output-scale-fit | electrode1-relative | 1-3-2-4 | absolute | flip-x | fitted / 0.81830778 | 10/10 | 6 | 1.83489 | 4.54058 |
| 35 | all-positive+output-scale-fit | electrode1-relative | 1-4-2-3 | absolute | rotate-180 | fitted / 0.80952336 | 10/10 | 9 | 1.84115 | 5.07789 |
| 36 | all-positive+output-scale-fit | absolute | 1-3-4-2 | absolute | swap-xy | fitted / 0.69196375 | 10/10 | 9 | 1.84559 | 5.47096 |
| 37 | basis-fitted-linear-combination+output-scale-fit | electrode1-relative | 1-3-2-4 | electrode1-relative | flip-x | fitted / 0.81594283 | 10/10 | 6 | 1.86161 | 4.63839 |
| 38 | all-positive+output-scale-fit | electrode1-relative | 1-3-4-2 | absolute | swap-xy | fitted / 0.68943233 | 10/10 | 8 | 1.86258 | 5.4025 |
| 39 | all-positive | electrode1-relative | 1-2-4-3 | absolute | swap-xy | none / 1 | 10/10 | 8 | 1.86382 | 5.4948 |
| 40 | all-positive+output-scale-fit | electrode1-relative | 1-2-4-3 | electrode1-relative | swap-xy | fitted / 0.80686184 | 10/10 | 8 | 1.87952 | 4.8618 |
| 41 | all-positive+output-scale-fit | absolute | 1-2-4-3 | electrode1-relative | swap-xy | fitted / 0.81360553 | 10/10 | 7 | 1.89001 | 5.04883 |
| 42 | all-positive+output-scale-fit | electrode1-relative | 1-3-2-4 | absolute | flip-x | fitted / 0.80212955 | 10/10 | 5 | 1.89034 | 4.51773 |
| 43 | basis-fitted-linear-combination | electrode1-relative | 1-3-2-4 | absolute | flip-x | none / 1 | 10/10 | 6 | 1.89465 | 5.01369 |
| 44 | all-positive+output-scale-fit | electrode1-relative | 1-3-4-2 | electrode1-relative | swap-xy | fitted / 0.68820909 | 10/10 | 8 | 1.89913 | 5.45059 |
| 45 | all-positive | absolute | 1-2-4-3 | electrode1-relative | swap-xy | none / 1 | 10/10 | 7 | 1.90819 | 5.59053 |
| 46 | all-positive+output-scale-fit | absolute | 1-3-4-2 | electrode1-relative | swap-xy | fitted / 0.68768383 | 10/10 | 9 | 1.90965 | 5.51526 |
| 47 | all-positive+output-scale-fit | electrode1-relative | 1-3-2-4 | electrode1-relative | flip-x | fitted / 0.80029845 | 10/10 | 5 | 1.91624 | 4.61699 |
| 48 | all-positive+output-scale-fit | absolute | 1-2-3-4 | absolute | swap-negated | fitted / 0.78422498 | 10/10 | 6 | 1.92262 | 5.39975 |
| 49 | basis-fitted-linear-combination | electrode1-relative | 1-3-2-4 | electrode1-relative | flip-x | none / 1 | 10/10 | 6 | 1.92856 | 5.14343 |
| 50 | all-positive+output-scale-fit | electrode1-relative | 1-3-4-2 | absolute | rotate-90 | fitted / 0.66968281 | 10/10 | 8 | 1.93582 | 4.99268 |

## Best hypothesis and mismatch classification

The best screening hypothesis was `basis_fitted_relative_perm1324_pppp__ref-absolute__identity__scale-none`:
mean `1.03867 mm`, maximum `5.90503 mm`, and exactly-three topology in `6/10`
rows.
After promotion, the best hypothesis is `basis_fitted_relative_perm1324_pppp__ref-absolute__identity__scale-fitted`:

- completed rows: `50/50`;
- exactly-three physical-minimum rows: `36/50`;
- mean error: `1.27046 mm`;
- median error: `1.05612 mm`;
- maximum error: `5.92593 mm`;
- p95 error: `2.45948 mm`;
- output scale: `0.925496632`;
- validation gate: `no`.

On the same rows 1--10, the best screen reduces the Milestone-5 mean from
1.08687 mm to 1.03867 mm, a `4.435%` improvement. The rows 1--50 promoted
mean is not directly comparable with Milestone 5's ten-row number. On rows
1--50 it improves over the promoted all-positive case (1.28335 mm) by only
`1.005%`.

Classification: **electrode numbering matters; global orientation is not the
cause; output scale is a minor correction; the basis fit converges to
all-positive, so polarity is not primary; the residual mismatch is primarily
model-class/topology limited**. The best all-positive screening mean was
1.05679 mm. The mixed relative-input/absolute-output-frame convention is a
small diagnostic gain, not a coherent physical explanation, and it does not
generalize to the validation gate.

## Decision

**NOT SAFE:** no promoted hypothesis meets the validation gate.
The gate requires every selected row to complete with exactly three
pre-selection Hessian-valid minima, mean error <=0.25 mm, and maximum
error <=0.5 mm. Diagnostic scale or voltage fitting does not change
the default physical model and cannot authorize generation by itself.
No ML or synthetic dataset generation was performed.

Total study runtime: `530.803 s`.
