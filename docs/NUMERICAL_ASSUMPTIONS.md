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
- The legacy six model inputs are `(dx2, dy2, dx3, dy3, dx4, dy4)` in metres
  and keep electrode 1 fixed. The main `Data.txt` validation input is instead
  the raw `(4,2)` array (or flat eight-vector): all four nominal electrode
  centres receive their absolute displacements while the outer boundary remains
  fixed at the origin. The old relative path remains available explicitly.
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
- The original recovered-gradient search remains the default and is available
  explicitly as `recovered-gradient`. `raw-element-diagnostic` reports local
  lows of the elementwise-constant P1 field and does not claim sub-element
  field zeros. `robust` is opt-in and does not change the potential solve.
- Robust candidate sources are the legacy coarse/refined candidates, analytic
  zeros of the recovered P1 vector field inside triangles, and raw-element
  local lows refined against the recovered field. Source locations within the
  existing merge tolerance are combined while retaining provenance and support
  count.
- Robust Hessians use four stencil lengths equal to 0.005, 0.0125, 0.025, and
  0.05 times the actual mesh parameter. All finite signatures must be positive,
  and the largest per-eigenvalue variation ratio must not exceed 8. A candidate
  within 0.02 mesh parameters of an internal facet is classified as
  facet-sensitive only if the adjacent raw-field jump is at least 0.5 of the
  larger adjacent magnitude and the Hessian is unstable.
- Robust recovered `|E|²` must be no more than 100 times the median scale of the
  lowest configured candidate set, with an explicit floating-point floor. All
  robust rejections and their individual reasons remain in diagnostics.
- `artifact_probability` is an uncalibrated rule score: 0.25 for close-facet
  location, 0.20 for a large adjacent raw-field jump, 0.40 for Hessian
  instability, and 0.15 for high recovered `|E|²`, capped at one. Low, medium,
  and high labels use 0.30 and 0.60 boundaries. No reference coordinates enter
  the score or acceptance decision.

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
- Milestone six screens all six E1-preserving E2--E4 source permutations, both
  displacement and minimum coordinate frames, the eight global symmetries of a
  square, positive output-scale fitting, all nontrivial binary voltage patterns
  modulo global sign, and a linear fit of four one-electrode basis fields. All
  variants remain diagnostic; the default all-positive physical model is not
  changed.
- The fitted basis vector is `(0.999926798, 0.999809418, 0.999786521, 1.0) V`,
  effectively all-positive. Alternate binary polarities are worse or fail. The
  best rows 1--50 diagnostic completes all rows but has exactly-three topology
  in only 36/50, 1.27046 mm mean error, and 5.92593 mm maximum error. The
  residual mismatch is classified primarily as model-class/topology limited,
  and synthetic dataset generation remains unsafe.
- Milestone seven validates the shared P1 Dirichlet solve against the analytic
  concentric-capacitor solution and an exactly linear potential. At h=2 mm the
  annulus potential relative L2 error is 1.61864e-4, recovered-field relative
  L2 error is 3.79513e-3, raw element-field relative L2 error is 2.81030e-2,
  and the free-node residual is 1.78918e-15. The linear test recovers
  `E=(-1,0)` to 7.74095e-15 V/m, confirming the `-grad(phi)` sign.
- Every real-scale h=2 mm boundary vertex is classified exactly once: 158 outer
  nodes and 32 nodes on each electrode, with zero missing or overlapping nodes
  and zero imposed-potential error. Nominal and reference rows 1--10 retain at
  least 9.60944 mm electrode separation and 17.6386 mm outer clearance.
- Coherent E2/E3-swapped validation mean errors are 1.08687, 1.08951, 1.07248,
  and 1.08454 mm at h=2.0, 1.5, 1.0, and 0.75 mm. Exactly-three counts are
  5/10, 9/10, 9/10, and 8/10. The optional rows 1--3 h=0.5 mm check reduces
  their h=2 mm mean by only 1.097%.
- The milestone-seven default artifact action is report-only `flag`. A candidate
  is flagged by a documented facet-lock plus adjacent raw-field-jump criterion,
  or by recovered `|E|^2` at least 100 times the row's best candidate. The CLI
  also supports explicit audit-only `filter`; neither action silently changes
  the forward API. At h=2 mm, 13/39 candidates and 4/30 selected minima are
  flagged.
- No assembly, sign, boundary-condition, or geometry bug was found. Error
  plateauing supports a likely model-class mismatch, but the selected artifact
  rate and non-monotone topology mean that the residual is not yet
  scientifically attributable to model class alone. Synthetic generation
  remains unsafe.
- Milestone eight reruns rows 1--10 with legacy, report-only audit flags, and
  robust selection. At h=2 mm under the best E1,E3,E2,E4 mapping, the legacy
  mean/maximum errors are 1.08687/5.98241 mm and exactly-three topology is 5/10.
  Robust results are 1.08754/5.98525 mm with exactly-three topology 10/10. It
  rejects 24 candidates and selects zero candidates that fail the robust
  multi-stencil/interpolation criteria.
- The four Milestone 7 conservative best-mapping flags remain reproducible in
  audit mode on rows 3, 4, 5, and 10, but their robust cell-zero counterparts
  have stable positive Hessians. The robust mean error worsens by only 0.061%,
  so those flags were not responsible for a significant fraction of the
  reference mismatch.
- Robust exactly-three topology is 10/10 at h=2.0, 1.5, 1.0, and 0.75 mm and
  3/3 at h=0.5 mm for rows 1--3. Of 99 successive-mesh branch matches, 96 are
  within 0.25 mm; all 69 transitions whose coarse mesh is 1.5 mm or finer pass,
  with maximum shift 0.164062 mm. Three 2.0-to-1.5 mm shifts reach 0.299800 mm.
- Robust post-processing therefore gives substantially stronger evidence that
  the remaining spatial mismatch belongs to the physical model class/topology,
  while not claiming exact mesh invariance. The validation gate still fails,
  so synthetic generation remains unsafe.
- Milestone nine adds an opt-in Gmsh background size field: 2 mm in the outer
  domain, 0.5 mm at electrode boundaries, and a configurable 8 mm-radius central
  disk. The legacy `MeshConfig` has no size field by default, so regression
  meshes and physical defaults are unchanged.
- The resumed local-refinement evidence completes rows 1--3 at 500 um with
  exactly-three robust topology and 0.913487 mm mean error. Only row 1 completes
  at 200 um; its mean error worsens by 0.791% and its maximum branch shift is
  30.1185 um. The interrupted 200 um rows 2--3 and 100 um attempts are retained
  as failures/timeouts. A 50 um all-modes pilot exceeded 600 s and was
  terminated. Finer disk-only preflight estimates rise from 1.16 million
  triangles at 20 um to 4.64 billion at 0.316228 um.
- Calibration screening is explicitly diagnostic. The best raw rows 1--3
  geometry uses 12 mm electrode radius and has 0.812039 mm mean error. The best
  fitted voltage vector is within 0.0205% of all-positive. Neither improvement
  generalizes enough to satisfy the reference gate.
- On rows 1--10 the 500 um central real-scale all-positive E1,E3,E2,E4 baseline
  has 1.082384 mm mean error. A bounded output scale/rotation/anisotropy fit lowers
  it only to 1.074555 mm and leaves 5.535433 mm maximum error. On rows 1--20 the
  untransformed baseline is best at 1.074730 mm mean, 5.970718 mm maximum, and
  exactly-three topology in 20/20 rows.
- With assembly, sign, boundary, geometry, postprocessing, local resolution,
  mapping, reasonable 2D geometry, and static voltage hypotheses audited, the
  leading missing assumption is the electrode/drive model: the reference
  article describes an eight-rod octupole while this FEM has four circular holes
  with one scalar potential per electrode. Synthetic generation remains unsafe.
