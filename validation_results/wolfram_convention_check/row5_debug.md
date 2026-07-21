# Wolfram-convention row 5 debug

Only `Data.txt` row 5 was solved. The transform is `F1,F2,F3,F4 = -[W3,W1,W4,W2]`; all four electrodes move and the 50 mm grounded outer circle remains fixed.

Mesh: 3992 nodes, 7376 triangles; relative residual 1.2188e-15; runtime 3.954 s.

## Displacements and positions

Raw `Data.txt` displacement pairs in Wolfram electrode order:

| Wolfram electrode | raw dx (m) | raw dy (m) |
|---:|---:|---:|
| W1 | 0.000280470407961 | 0.000228510371172 |
| W2 | 0.000229628862973 | 0.000185888126092 |
| W3 | -3.88415228379e-05 | -0.000400668295176 |
| W4 | 0.000347220199041 | -0.000352798004149 |

Transformed absolute pairs applied in FEM electrode order:

| FEM electrode | source | transformed dx (m) | transformed dy (m) |
|---:|---:|---:|---:|
| F1 | -W3 | 3.88415228379e-05 | 0.000400668295176 |
| F2 | -W1 | -0.000280470407961 | -0.000228510371172 |
| F3 | -W4 | -0.000347220199041 | 0.000352798004149 |
| F4 | -W2 | -0.000229628862973 | -0.000185888126092 |

| index | reference x (mm) | reference y (mm) | computed x (mm) | computed y (mm) | computed |E|^2 (V^2/m^2) |
|---:|---:|---:|---:|---:|---:|
| 1 | -2.31 | -1.39 | -0.728356122 | 3.58380993 | 7.99336159e-37 |
| 2 | 2.66 | -1.51 | -2.35499375 | -1.41498853 | 6.5122387e-36 |
| 3 | 2.57 | -1.55 | 2.57746242 | -1.60615516 | 1.62805968e-35 |

## All robust candidates before final selection

| id | sources | x (mm) | y (mm) | recovered |E|^2 | accepted | selected | interpolation-sensitive | artifact probability | classification / reason |
|---:|---|---:|---:|---:|---|---|---|---:|---|
| 1 | raw-element-local-low,recovered-cell-zero,recovered-coarse | -0.728356122 | 3.58380993 | 7.99336159e-37 | True | True | False | 0 | low: accepted-stable-low-psi |
| 2 | raw-element-local-low,recovered-cell-zero,recovered-coarse | -2.35499375 | -1.41498853 | 6.5122387e-36 | True | True | False | 0.2 | low: accepted-stable-low-psi |
| 3 | raw-element-local-low,recovered-cell-zero,recovered-coarse | 2.57746242 | -1.60615516 | 1.62805968e-35 | True | True | False | 0.45 | medium: accepted-stable-low-psi |

Rejected candidates: 0.

## Matching diagnostics

Pairwise distances in millimetres (rows R1--R3, columns C1--C3):

| | C1 | C2 | C3 |
|---|---:|---:|---:|
| R1 | 5.21923199 | 0.0514671192 | 4.89223998 |
| R2 | 6.11783106 | 5.01589369 | 0.126721216 |
| R3 | 6.10206174 | 4.92684397 | 0.0566488284 |

| rank | reference -> computed | errors (mm) | total (mm) | maximum (mm) |
|---:|---|---|---:|---:|
| 1 | (2, 1, 3) | 0.0514671192, 6.11783106, 0.0566488284 | 6.225947 | 6.11783106 |
| 2 | (2, 3, 1) | 0.0514671192, 0.126721216, 6.10206174 | 6.28025008 | 6.10206174 |
| 3 | (1, 3, 2) | 5.21923199, 0.126721216, 4.92684397 | 10.2727972 | 5.21923199 |
| 4 | (1, 2, 3) | 5.21923199, 5.01589369, 0.0566488284 | 10.2917745 | 5.21923199 |
| 5 | (3, 1, 2) | 4.89223998, 6.11783106, 4.92684397 | 15.936915 | 6.11783106 |
| 6 | (3, 2, 1) | 4.89223998, 5.01589369, 6.10206174 | 16.0101954 | 6.10206174 |

Hungarian assignment: (2, 1, 3); errors [0.051467119203981226, 6.117831055858265, 0.05664882838309383] mm.
Direct nearest-neighbor indices: (2, 3, 3); errors [0.051467119203981226, 0.1267212162779357, 0.05664882838309383] mm. Duplicate computed choices: True.

Best coordinate-only post-transform checks:

| transform | mean (mm) | maximum (mm) |
|---|---:|---:|
| flip-x | 2.01517839 | 5.44776539 |
| identity | 2.07531567 | 6.11783106 |
| rotate-90 | 2.35479523 | 4.22112994 |
| swap-negated | 2.64706727 | 4.06057019 |
| swap-xy | 2.79930626 | 5.87164525 |
| rotate-180 | 2.89754797 | 3.00806946 |
| rotate-270 | 3.1270775 | 5.57934672 |
| flip-y | 3.26586198 | 3.87498851 |

## Diagnosis

- Matching bug: **no**. The Hungarian result is the minimum-total-distance permutation among all 3! assignments.
- Direct-nearest-neighbor collision: **yes**. References (2, 3) are only 0.0984886 mm apart and both choose computed minimum C3; a one-to-one assignment cannot reuse that minimum.
- Missing distinct FEM branch for the close reference pair: **yes**. The nearest retained candidate to the Hungarian-outlier reference R2 is candidate 3 at 0.126721 mm, but that same candidate is needed by the other member of the close reference pair. No second candidate is present there.
- Selected spurious/interpolation-sensitive minimum: **no**. The distant upper branch is accepted, stable, low-|E|^2, and has no interpolation-sensitive flag.
- Robust topology-count failure: **no**; three candidates are selected and 0 are retained as rejected diagnostics.
- Coordinate sign/order issue: **no evidence**. Every tested coordinate-only transform retains a multi-millimetre maximum error.

Overall classification: **real row-specific FEM/reference branch/topology mismatch, not a matching bug**. The reference has two distinct minima in one tight cluster, while this FEM solve has only one minimum in that cluster plus a numerically robust upper branch. There are no rejected candidates to recover.

## Gate sensitivity

Removing row 5 leaves 9/9 exactly-three rows, mean 0.0472195051 mm and maximum 0.128333847 mm. The validation gate would **pass** on those nine rows.
Removing only the single 6.11783106 mm outlier leaves 29 matches with mean 0.0476911236 mm and maximum 0.128333847 mm.
Replacing that outlier by an error at the 0.5 mm gate limit, with every other error unchanged, gives mean 0.0627680862 mm and maximum 0.5 mm, so the aggregate gate would pass.
