# Focused investigation: fourth Hessian-valid candidate

## Case

- Mesh characteristic length: `60 µm`
- Outer radius: `4 mm`
- Nodes / triangles: `16304` / `32059`
- Relative free-node residual: `2.604181e-15`
- Hessian-valid before selection: `4`
- Returned by forward API: `3`

## All pre-selection candidates

Distances to electrodes are surface clearances. The search boundary is the
configured square used by the optimizer, not the outer electrostatic boundary.

| rank by Ψ | x (µm) | y (µm) | Ψ (V²/m²) | Hessian λ1 (V²/m⁴) | Hessian λ2 (V²/m⁴) | nearest electrode | electrode clearance (µm) | search clearance (µm) | nearest candidate (µm) | nearest mesh facet (µm) | selected |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 289.34077 | 257.03309 | 2.51064901e-04 | 2.96994672e+10 | 3.31098198e+10 | 2 | 461.53362 | 410.65923 | 259.3298 | 1.5231508 | yes |
| 2 | 322.12902 | -217.68803 | 4.88989949e-04 | 3.85275853e+10 | 4.38062848e+10 | 4 | 438.69523 | 377.87098 | 225.322 | 1.1082379 | yes |
| 3 | -471.26561 | 98.666851 | 7.91188772e-04 | 7.37145594e+10 | 9.78857100e+10 | 3 | 458.81685 | 228.73439 | 739.17063 | 5.5526787 | yes |
| 4 | 261.18694 | -0.76393173 | 1.94465258e+02 | 3.43943533e+10 | 1.58344322e+11 | 1 | 518.8134 | 438.81306 | 225.322 | 0.011112234 | no |

## Fourth-candidate checks

- Candidate coordinates: `(261.186943, -0.763932) µm`.
- Its Ψ is `245789×` the largest selected-candidate Ψ; it is not null-like.
- Nearest electrode and search-window clearances are `518.813 µm` and `438.813 µm`.
- Nearest other candidate is `225.322 µm` away; the configured merge threshold is `20 µm`.
- The optimizer stopped `0.011112 µm` from a mesh facet (`0.000185204` mesh lengths).
- The adjacent raw P1 element fields differ by `0.123392 V/m`.
- Their magnitudes are `14.3012` and `14.1783 V/m`; the recovered candidate magnitude is `13.9451 V/m`, so none is a field null.
- The smallest-stencil to largest-positive-stencil Hessian λ1 ratio is `160.164`; a smooth Hessian should approach a finite value as the stencil shrinks.
- Perturbed meshes (59 µm, 61 µm) contain only the expected three candidates: **yes**.

### Hessian stencil sensitivity

| step (µm) | λ1 (V²/m⁴) | λ2 (V²/m⁴) | positive definite |
|---:|---:|---:|:---:|
| 1 | 1.32228101e+11 | 6.26702922e+11 | yes |
| 2 | 6.78940775e+10 | 3.15352325e+11 | yes |
| 4 | 3.43943533e+10 | 1.58344322e+11 | yes |
| 8 | 1.73113067e+10 | 7.95071508e+10 | yes |
| 16 | 8.93109566e+09 | 3.97606642e+10 | yes |
| 24 | 8.25578155e+08 | 2.64809863e+10 | yes |
| 32 | -4.68357094e+09 | 2.10409574e+10 | no |

### Local mesh-size perturbation

| h (µm) | nodes | triangles | Hessian-valid candidates |
|---:|---:|---:|---:|
| 59 | 17134 | 33708 | 3 |
| 60 | 16304 | 32059 | 4 |
| 61 | 15671 | 30803 | 3 |

## Classification

**Likely cause: recovered-gradient interpolation artifact at a mesh facet.**

- Physical minimum likely: **no**
- Boundary/search artifact: **no**
- Duplicate/merge-threshold issue: **no**
- Recovered-gradient interpolation artifact: **yes**

For a smooth two-dimensional harmonic potential, a nonconstant analytic electric
field cannot have a strict interior local minimum of its magnitude. Here the
candidate has a large nonzero field magnitude, lies essentially on a recovered-P1
facet, disappears after ±1 µm mesh perturbations, and its finite-difference
curvature grows strongly as the stencil shrinks. Together these diagnose a
piecewise-interpolation kink that passes the configured Hessian test, not a
fourth physical RF null.

No physical equation, boundary condition, or forward-selection rule was changed.
