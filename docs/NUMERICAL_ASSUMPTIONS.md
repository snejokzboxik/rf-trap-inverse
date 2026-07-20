# Numerical assumptions and limitations

## Physical model

- The calculation is a two-dimensional cross-section of infinitely long,
  perfectly conducting cylindrical electrodes; end effects are absent.
- All four electrodes have one normalized Dirichlet value (`+1 V` in the
  demonstrator), and the outer boundary has `0 V`.
- The outer boundary is a circle centred at the coordinate origin.  It replaces
  infinity at a finite, configurable radius; outer-radius convergence has not
  yet been established.
- Electrode 1 is fixed.  The six model inputs are `(dx2, dy2, dx3, dy3, dx4,
  dy4)` in metres and are added to configurable nominal centres.
- `|E|²` is only proportional to the physical RF pseudopotential.  No ion charge,
  mass, drive frequency, or amplitude scaling is included.
- The nominal radius and positions in the example/tests are provisional values
  chosen solely to exercise the pipeline.  They are not experimental inputs.

## Discretization

- Gmsh's OpenCASCADE Boolean difference creates a disk with four exact circular
  holes, then a first-order triangular mesh approximates every curve by chords.
- The mesh is globally controlled by one characteristic length.  No adaptive
  error estimator or boundary-layer refinement is implemented in milestone 1.
- The potential uses continuous piecewise-linear (`P1`) finite elements.  The
  weak problem is the standard Galerkin discretization of Laplace's equation.
- Gmsh is restricted to one thread and a fixed seed.  Reproducibility is expected
  for the pinned major-version range and platform, but byte-identical meshes are
  not promised across Gmsh versions or operating systems.

## Field and minima

- A `P1` potential has an elementwise-constant, facet-discontinuous gradient.
  To make local optimization meaningful, element gradients are averaged at each
  vertex with triangle-area weights and the nodal vectors are interpolated with
  a continuous `P1` field.  This is a gradient-recovery approximation and is the
  principal post-processing error source.
- The minima search covers a configurable square centred at the origin.  The
  caller must choose an extent containing the physical nulls of interest.
- A Cartesian coarse scan supplies candidates.  Candidate refinement uses
  L-BFGS-B in coordinates scaled by the search half-extent.  The box constraint is
  exact; points outside the vacuum domain receive a finite penalty and are
  rejected after refinement.
- Refined points closer than a configurable Euclidean tolerance are merged,
  retaining the lower `|E|²` value.
- Hessians of `|E|²` are central finite differences on a configurable stencil.
  Both eigenvalues must exceed the configured threshold.
- If more than three validated candidates remain, the three lowest `|E|²` values
  are selected.  Fewer than three is an explicit error rather than silently
  fabricating or padding outputs.
- Final ordering uses polar angle in `[0, 2π)` about the coordinate origin.

## Validation status

- Unit tests cover displacement application and geometry rejection, boundary
  conformance, Dirichlet enforcement, the discrete maximum principle, the free
  linear-system residual, exact recovery of a synthetic linear field, and the
  full demonstrator configuration.
- A production accuracy claim requires mesh-size and outer-radius convergence
  studies against the supplied physical geometry.  Those studies are deliberately
  left for the next forward-model validation milestone.

