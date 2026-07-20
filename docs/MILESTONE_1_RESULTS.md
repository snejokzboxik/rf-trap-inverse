# Milestone 1 verification results

Verification date: 2026-07-20  
Platform: Windows, Python 3.12.13

Installed numerical versions used for this verification:

- Gmsh 4.15.2
- NumPy 2.5.1
- SciPy 1.18.0
- scikit-fem 12.0.2
- pytest 9.1.1

## One-configuration result

The provisional nominal geometry and the displacement vector are defined in
`examples/run_one_configuration.py`.  The displacement input in micrometres was:

```text
(120, -80, -150, 110, 90, 160)
```

With a mesh characteristic length of `80 µm`, the run produced:

- 9,537 vertices and 18,661 triangles;
- relative free-node linear residual: `2.396e-15`;
- 4,973 valid coarse points;
- 3 coarse candidates, 3 refined candidates, 3 unique candidates, and 3
  positive-Hessian candidates;
- successful L-BFGS-B termination for all three refinements.

Angle-sorted results:

| Minimum | x (µm) | y (µm) | `|E|²` (V²/m²) | Hessian eigenvalues (V²/m⁴) |
|---:|---:|---:|---:|---:|
| 1 | 289.4057 | 255.8637 | 2.102560e-4 | 1.730424e10, 2.554580e10 |
| 2 | -472.2072 | 101.2972 | 1.092595e-3 | 7.305053e10, 1.058532e11 |
| 3 | 320.8861 | -217.8246 | 1.336463e-5 | 3.470670e10, 5.659876e10 |

`|E|²` is the normalized pseudopotential proxy, not an energy.

## Automated checks

Eight tests passed in 3.56 seconds on the verification machine.  They cover:

- fixed-reference displacement semantics and invalid-geometry rejection;
- analytic vacuum-membership rejection of electrode interiors;
- nonempty conforming mesh and complete boundary-node classification;
- exact reproduction of mesh coordinates and connectivity on a repeated run with
  the same seed in the same environment;
- exact Dirichlet values, discrete maximum principle, and linear residual;
- exact recovered field for a globally linear synthetic potential;
- three valid, positive-Hessian, polar-angle-sorted end-to-end minima.

`python -m compileall` also completed without errors.

## Mesh-sensitivity smoke check

The same configuration was evaluated at characteristic lengths of 120, 80, and
60 µm.  The numbers of vertices were 4,233, 9,537, and 16,304, respectively.
All runs found the same three angle-ordered minima.  The maximum positional shift
from 120 to 80 µm was approximately 3.66 µm; from 80 to 60 µm it was approximately
2.79 µm.

This is evidence that the pipeline behaves consistently for the demonstrator,
not a production convergence claim.  Formal outer-radius and mesh-convergence
criteria must be established after the physical nominal geometry is supplied.

