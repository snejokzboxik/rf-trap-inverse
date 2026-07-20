# Numerical assumptions and limitations

## Physical model

- The calculation is a two-dimensional cross-section of infinitely long,
  perfectly conducting cylindrical electrodes; end effects are absent.
- The physical-model default keeps all four electrodes at the same normalized
  Dirichlet value (`+1 V`) and the outer boundary at `0 V`. The solver now also
  accepts an explicit four-value tuple solely for named diagnostic polarity
  cases; this does not silently change the default.
- The outer boundary is a circle centred at the coordinate origin. It replaces
  infinity at a finite, configurable radius. The real-scale reference-validation
  configuration uses 50 mm. Earlier 3.5, 4.0, and 5.0 mm studies apply only to
  the retained demonstrator.
- Electrode 1 is fixed. The six model inputs are `(dx2, dy2, dx3, dy3, dx4,
  dy4)` in metres and are added to configurable nominal centres. An
  absolute-frame validation diagnostic applies raw E1 displacement by creating a
  row-specific geometry; the core six-coordinate API remains unchanged.
- `|E|²` is only proportional to the physical RF pseudopotential.  No ion charge,
  mass, drive frequency, or amplitude scaling is included.
- The example and milestone-one/two regression tests retain provisional values.
  Reference validation instead uses 10 mm electrode radius, 11.48 mm inner
  surface radius, 21.48 mm electrode-centre radius, and a 50 mm outer boundary.
  With `a=21.48 mm/sqrt(2)`, numbering is E1 `(-a,+a)`, E2 `(+a,+a)`, E3
  `(-a,-a)`, and E4 `(+a,-a)`.

## Discretization

- Gmsh's OpenCASCADE Boolean difference creates a disk with four exact circular
  holes, then a first-order triangular mesh approximates every curve by chords.
- The mesh is globally controlled by one characteristic length.  No adaptive
  error estimator or boundary-layer refinement is implemented in milestone 1.
- The potential uses continuous piecewise-linear (`P1`) finite elements.  The
  weak problem is the standard Galerkin discretization of Laplace's equation.
- Gmsh is restricted to one thread, a positive fixed seed, explicit OpenCASCADE
  primitive tags, and its reproducibility mode. Python command-line arguments and
  user Gmsh configuration files are excluded from initialization. Each production
  convergence case runs in a fresh interpreter because Gmsh can retain meshing
  state within a long-lived process. Byte-identical meshes are still not promised
  across Gmsh versions or operating systems.

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
- `MinimaDiagnostics.hessian_validated_minima` retains every positive-Hessian
  candidate before the lowest-three selection so numerical topology can be
  audited without changing the forward API output.
- Final ordering uses polar angle in `[0, 2π)` about the coordinate origin.

## Convergence reporting

- The configured mesh sizes and outer radii form a full Cartesian product. This
  is more expensive than two isolated sweeps but exposes interactions between the
  discretization scale and finite outer boundary.
- Stored minima retain the forward API's polar-angle ordering. Successive runs
  are compared using minimum-total-distance bipartite assignment, which avoids a
  false jump if polar ordering changes.
- Mesh refinement is ordered from larger to smaller characteristic length.
  Outer-radius comparison is ordered from smaller to larger radius.
- “Three-minimum structure stable” requires exactly three positive-Hessian
  candidates before final selection in every run, as well as three returned
  minima. A user-configured coordinate tolerance is reported separately and does
  not redefine topology.
- Plot coordinates are displayed in micrometres for readability; CSV storage and
  all calculations remain in SI units.

## Validation status

- Unit tests cover displacement application and geometry rejection, boundary
  conformance, Dirichlet enforcement, the discrete maximum principle, the free
  linear-system residual, exact recovery of a synthetic linear field, and the
  full demonstrator configuration. Synthetic and mocked tests additionally cover
  factorial sweep construction, spatial matching, stability decisions, CSV and
  Markdown serialization, and headless plot generation.
- Milestone two performed the provisional 3×3 numerical study. One run produced
  four Hessian-valid recovered-field candidates before selecting the required
  three outputs. Focused follow-up found that the fourth point is 0.0111 µm from
  an internal mesh facet, has a large nonzero `|E|²`, disappears for 59 and 61 µm
  meshes, and has stencil-dependent curvature. It is therefore classified as a
  recovered-gradient interpolation artifact rather than a fourth physical null.
- A production accuracy claim still requires convergence criteria applied to the
  supplied physical geometry. The current Hessian filter is intentionally left
  unchanged; the focused diagnostics make its facet-kink failure mode visible.
- Milestone four compared reference rows 1–10 in the electrode-1-relative frame.
  Eight rows returned three selected minima, but only seven had exactly three
  pre-selection Hessian-valid candidates. Across 24 assignment-matched minima,
  the median error was 3.08938 mm and the maximum was 4.93435 mm.
- The reference benchmark is outside the current demonstrator's validated scale:
  29 of 30 reference minima are outside the ±0.7 mm search square and 13 are
  outside the 4.0 mm outer circle. The current provisional geometry is therefore
  not validated against the supplied data, and synthetic dataset generation is
  not yet justified.
- Milestone five uses the real 50 mm geometry and an ±8 mm search square, so all
  reference minima are within the intended search scale. Identity-numbering
  all-positive runs complete all ten rows at 2.0, 1.5, and 1.0 mm mesh sizes;
  mean errors decrease only from 1.47368 to 1.44011 mm. Strict exactly-three
  topology improves from 5/10 to 9/10 rows but is not stable across all runs.
- The best tested convention exchanges source electrodes 2 and 3 while retaining
  the FEM numbering above. At 2.0 mm it gives 1.08687 mm mean, 0.953208 mm
  median, and 5.98241 mm maximum error, with exactly-three topology in only 5/10
  rows. Both alternating-polarity coordinate-frame variants find one validated
  minimum per row rather than three. The model is closer than Milestone 4 but is
  still not validated for synthetic dataset generation.
