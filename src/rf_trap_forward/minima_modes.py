"""Explicit legacy, raw-element, and robust minima post-processing modes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import MinimaSearchConfig
from .field import RecoveredField, element_electric_fields
from .mesh import nearest_internal_mesh_facet
from .minima import (
    LocalMinimum,
    MinimaDiagnostics,
    collect_recovered_candidates,
    find_local_minima,
    finite_difference_hessian,
    refine_recovered_candidate,
)

MinimaMode = Literal[
    "recovered-gradient",
    "raw-element-diagnostic",
    "robust",
]


@dataclass(frozen=True)
class RobustMinimaConfig:
    """Documented numerical controls for robust candidate classification."""

    hessian_step_mesh_fractions: tuple[float, ...] = (0.005, 0.0125, 0.025, 0.05)
    maximum_hessian_eigenvalue_variation_ratio: float = 8.0
    internal_facet_distance_mesh_fraction: float = 0.02
    adjacent_raw_field_jump_ratio: float = 0.50
    maximum_recovered_psi_ratio: float = 100.0
    raw_element_source_limit: int = 24

    def __post_init__(self) -> None:
        """Validate every robust-search threshold."""

        values = np.asarray(
            (
                *self.hessian_step_mesh_fractions,
                self.maximum_hessian_eigenvalue_variation_ratio,
                self.internal_facet_distance_mesh_fraction,
                self.adjacent_raw_field_jump_ratio,
                self.maximum_recovered_psi_ratio,
            ),
            dtype=float,
        )
        if (
            len(self.hessian_step_mesh_fractions) < 3
            or not np.all(np.isfinite(values))
            or np.any(values <= 0.0)
        ):
            raise ValueError("robust minima thresholds must be finite and positive")
        if len(set(self.hessian_step_mesh_fractions)) != len(
            self.hessian_step_mesh_fractions
        ):
            raise ValueError("Hessian stencil fractions must be unique")
        if self.raw_element_source_limit <= 0:
            raise ValueError("raw_element_source_limit must be positive")


@dataclass(frozen=True)
class CandidateQualityMetrics:
    """Mesh, field, Hessian, and rule-based quality diagnostics for one candidate."""

    candidate_id: int
    source_names: tuple[str, ...]
    source_support_count: int
    position_m: NDArray[np.float64]
    recovered_psi_v2_per_m2: float
    recovered_field_magnitude_v_per_m: float
    containing_raw_field_magnitude_v_per_m: float
    adjacent_raw_field_magnitudes_v_per_m: tuple[float, float]
    adjacent_raw_field_jump_v_per_m: float
    adjacent_raw_field_jump_ratio: float
    distance_to_internal_facet_m: float
    distance_to_internal_facet_over_mesh_h: float
    distance_to_nearest_electrode_m: float
    hessian_steps_m: tuple[float, ...]
    hessian_eigenvalues_v2_per_m4: tuple[tuple[float, float], ...]
    hessian_stability_classification: str
    hessian_stable: bool
    hessian_eigenvalue_variation_ratio: float
    recovered_psi_ratio_to_candidate_scale: float
    physically_small_recovered_psi: bool
    facet_sensitive: bool
    legacy_interpolation_flag: bool
    interpolation_sensitive: bool
    artifact_probability: float
    artifact_classification: str
    classification_reason: str
    robust_accepted: bool
    selected: bool = False


@dataclass(frozen=True)
class MinimaModeResult:
    """Selected minima and complete candidate diagnostics for one mode."""

    mode: MinimaMode
    minima: tuple[LocalMinimum, ...]
    candidates: tuple[CandidateQualityMetrics, ...]
    legacy_diagnostics: MinimaDiagnostics | None
    expected_minima: int = 3

    @property
    def completed(self) -> bool:
        """Return whether the mode selected the configured three outputs."""

        return len(self.minima) == self.expected_minima

    @property
    def accepted_candidates(self) -> int:
        """Return the number of candidates accepted by this mode."""

        return sum(item.robust_accepted for item in self.candidates)

    @property
    def rejected_candidates(self) -> int:
        """Return the number of candidates retained as rejected diagnostics."""

        return len(self.candidates) - self.accepted_candidates

    @property
    def selected_interpolation_sensitive(self) -> int:
        """Return the selected-candidate count carrying an artifact flag."""

        return sum(item.selected and item.interpolation_sensitive for item in self.candidates)


@dataclass(frozen=True)
class CandidateSeed:
    """One candidate location with merged provenance from one or more sources."""

    position_m: NDArray[np.float64]
    source_names: tuple[str, ...]
    source_support_count: int
    optimizer_succeeded: bool

    def __post_init__(self) -> None:
        """Validate the public candidate seed and copy its position array."""

        position = np.asarray(self.position_m, dtype=float)
        if position.shape != (2,) or not np.all(np.isfinite(position)):
            raise ValueError("position_m must be one finite two-dimensional point")
        if not self.source_names or self.source_support_count <= 0:
            raise ValueError("candidate provenance and support count are required")
        object.__setattr__(self, "position_m", position.copy())


def classify_hessian_stability(
    eigenvalues_v2_per_m4: ArrayLike,
    *,
    maximum_variation_ratio: float = 8.0,
    minimum_eigenvalue: float = 0.0,
) -> tuple[str, bool, float]:
    """Classify multi-stencil Hessian signatures and eigenvalue variation."""

    values = np.asarray(eigenvalues_v2_per_m4, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] < 3:
        raise ValueError("eigenvalues_v2_per_m4 must have shape (n>=3, 2)")
    if not np.isfinite(maximum_variation_ratio) or maximum_variation_ratio <= 1.0:
        raise ValueError("maximum_variation_ratio must exceed one")
    valid = np.all(np.isfinite(values), axis=1)
    if np.count_nonzero(valid) < 3:
        return "insufficient-valid-stencils", False, float("inf")
    finite = values[valid]
    if np.any(finite <= minimum_eigenvalue):
        return "unstable-hessian-signature", False, float("inf")
    variations = np.max(finite, axis=0) / np.min(finite, axis=0)
    variation = float(np.max(variations))
    if variation > maximum_variation_ratio:
        return "unstable-hessian-magnitude", False, variation
    return "stable-positive-hessian", True, variation


def classify_facet_sensitive(
    distance_over_mesh_h: float,
    adjacent_raw_field_jump_ratio: float,
    hessian_stable: bool,
    *,
    distance_threshold: float = 0.02,
    jump_threshold: float = 0.50,
) -> bool:
    """Return whether facet proximity, field jump, and Hessian instability coincide."""

    return bool(
        np.isfinite(distance_over_mesh_h)
        and np.isfinite(adjacent_raw_field_jump_ratio)
        and distance_over_mesh_h <= distance_threshold
        and adjacent_raw_field_jump_ratio >= jump_threshold
        and not hessian_stable
    )


def run_minima_mode(
    recovered_field: RecoveredField,
    potential_v: ArrayLike,
    search_config: MinimaSearchConfig,
    *,
    mode: MinimaMode = "recovered-gradient",
    robust_config: RobustMinimaConfig | None = None,
) -> MinimaModeResult:
    """Run one explicit minima mode without changing the physical FEM solution."""

    controls = robust_config or RobustMinimaConfig()
    if mode == "recovered-gradient":
        minima, diagnostics = find_local_minima(recovered_field, search_config)
        seeds = tuple(
            CandidateSeed(
                item.position_m,
                ("recovered-coarse",),
                1,
                item.optimizer_succeeded,
            )
            for item in diagnostics.hessian_validated_minima
        )
        candidates = _quality_table(
            seeds,
            recovered_field,
            potential_v,
            search_config,
            controls,
            accept_by_robust_rules=False,
        )
        return MinimaModeResult(
            mode=mode,
            minima=minima,
            candidates=_mark_selected(candidates, minima),
            legacy_diagnostics=diagnostics,
            expected_minima=search_config.expected_minima,
        )
    if mode == "raw-element-diagnostic":
        return _run_raw_element_mode(
            recovered_field,
            potential_v,
            search_config,
            controls,
        )
    if mode == "robust":
        return _run_robust_mode(
            recovered_field,
            potential_v,
            search_config,
            controls,
        )
    raise ValueError(f"unsupported minima mode: {mode}")


def select_robust_candidates(
    candidates: Sequence[CandidateQualityMetrics],
    expected_minima: int,
) -> tuple[LocalMinimum, ...]:
    """Select stable candidates deterministically while preserving rejections."""

    if expected_minima <= 0:
        raise ValueError("expected_minima must be positive")
    accepted = [item for item in candidates if item.robust_accepted]
    ordered = sorted(
        accepted,
        key=lambda item: (
            item.artifact_probability,
            item.recovered_psi_v2_per_m2,
            float(item.position_m[0]),
            float(item.position_m[1]),
        ),
    )[:expected_minima]
    minima = tuple(
        LocalMinimum(
            position_m=item.position_m.copy(),
            pseudopotential_v2_per_m2=item.recovered_psi_v2_per_m2,
            hessian_eigenvalues_v2_per_m4=np.asarray(
                item.hessian_eigenvalues_v2_per_m4[
                    len(item.hessian_eigenvalues_v2_per_m4) // 2
                ],
                dtype=float,
            ),
            optimizer_succeeded=True,
        )
        for item in ordered
    )
    return tuple(sorted(minima, key=lambda item: item.polar_angle_rad))


def _run_robust_mode(
    recovered_field: RecoveredField,
    potential_v: ArrayLike,
    search_config: MinimaSearchConfig,
    controls: RobustMinimaConfig,
) -> MinimaModeResult:
    collection = collect_recovered_candidates(recovered_field, search_config)
    seeds = [
        CandidateSeed(
            item.position_m,
            ("recovered-coarse",),
            1,
            item.optimizer_succeeded,
        )
        for item in collection.unique_candidates
    ]
    seeds.extend(_cell_zero_seeds(recovered_field, search_config))
    for raw_seed in _raw_element_seeds(
        recovered_field,
        potential_v,
        search_config,
        controls.raw_element_source_limit,
    ):
        refined = refine_recovered_candidate(
            recovered_field,
            raw_seed.position_m,
            search_config,
        )
        if refined is not None:
            seeds.append(
                    CandidateSeed(
                    refined.position_m,
                    ("raw-element-local-low",),
                    1,
                    refined.optimizer_succeeded,
                )
            )
    merged = _merge_seed_sources(
        seeds,
        recovered_field,
        search_config.merge_distance_m,
    )
    candidates = _quality_table(
        merged,
        recovered_field,
        potential_v,
        search_config,
        controls,
        accept_by_robust_rules=True,
    )
    minima = select_robust_candidates(candidates, search_config.expected_minima)
    diagnostics = MinimaDiagnostics(
        valid_coarse_points=collection.valid_coarse_points,
        coarse_candidates=collection.coarse_candidates,
        refined_candidates=collection.refined_candidates,
        unique_candidates=len(collection.unique_candidates),
        hessian_validated_candidates=len(collection.hessian_validated_minima),
        hessian_validated_minima=collection.hessian_validated_minima,
    )
    return MinimaModeResult(
        mode="robust",
        minima=minima,
        candidates=_mark_selected(candidates, minima),
        legacy_diagnostics=diagnostics,
        expected_minima=search_config.expected_minima,
    )


def _run_raw_element_mode(
    recovered_field: RecoveredField,
    potential_v: ArrayLike,
    search_config: MinimaSearchConfig,
    controls: RobustMinimaConfig,
) -> MinimaModeResult:
    seeds = _raw_element_seeds(
        recovered_field,
        potential_v,
        search_config,
        controls.raw_element_source_limit,
    )
    raw_fields = element_electric_fields(recovered_field.mesh, potential_v)
    finder = recovered_field.mesh.element_finder()
    records = []
    for index, seed in enumerate(seeds, start=1):
        triangle = int(finder(seed.position_m[0:1], seed.position_m[1:2])[0])
        magnitude = float(np.linalg.norm(raw_fields[:, triangle]))
        records.append(
            CandidateQualityMetrics(
                candidate_id=index,
                source_names=seed.source_names,
                source_support_count=1,
                position_m=seed.position_m.copy(),
                recovered_psi_v2_per_m2=float(
                    recovered_field.pseudopotential(seed.position_m)
                ),
                recovered_field_magnitude_v_per_m=float(
                    np.sqrt(recovered_field.pseudopotential(seed.position_m))
                ),
                containing_raw_field_magnitude_v_per_m=magnitude,
                adjacent_raw_field_magnitudes_v_per_m=(float("nan"), float("nan")),
                adjacent_raw_field_jump_v_per_m=float("nan"),
                adjacent_raw_field_jump_ratio=float("nan"),
                distance_to_internal_facet_m=float("nan"),
                distance_to_internal_facet_over_mesh_h=float("nan"),
                distance_to_nearest_electrode_m=_electrode_clearance(
                    recovered_field, seed.position_m
                ),
                hessian_steps_m=(),
                hessian_eigenvalues_v2_per_m4=(),
                hessian_stability_classification="not-applicable-raw-element",
                hessian_stable=False,
                hessian_eigenvalue_variation_ratio=float("nan"),
                recovered_psi_ratio_to_candidate_scale=float("nan"),
                physically_small_recovered_psi=False,
                facet_sensitive=False,
                legacy_interpolation_flag=False,
                interpolation_sensitive=False,
                artifact_probability=0.0,
                artifact_classification="diagnostic-only",
                classification_reason="raw-element-local-low-representative",
                robust_accepted=True,
            )
        )
    ordered = sorted(records, key=lambda item: item.containing_raw_field_magnitude_v_per_m)
    selected_records = ordered[: search_config.expected_minima]
    minima = tuple(
        sorted(
            (
                LocalMinimum(
                    item.position_m.copy(),
                    item.containing_raw_field_magnitude_v_per_m**2,
                    np.asarray([np.nan, np.nan]),
                    True,
                )
                for item in selected_records
            ),
            key=lambda item: item.polar_angle_rad,
        )
    )
    return MinimaModeResult(
        mode="raw-element-diagnostic",
        minima=minima,
        candidates=_mark_selected(tuple(records), minima),
        legacy_diagnostics=None,
        expected_minima=search_config.expected_minima,
    )


def _cell_zero_seeds(
    recovered_field: RecoveredField,
    search_config: MinimaSearchConfig,
) -> tuple[CandidateSeed, ...]:
    mesh = recovered_field.mesh
    nodal = recovered_field.electric_field_nodes_v_per_m
    seeds = []
    for triangle in mesh.t.T:
        values = nodal[:, triangle]
        matrix = np.column_stack((values[:, 1] - values[:, 0], values[:, 2] - values[:, 0]))
        determinant = float(np.linalg.det(matrix))
        scale = max(float(np.linalg.norm(matrix, ord=2)), 1.0)
        if abs(determinant) <= np.finfo(float).eps * scale * scale:
            continue
        uv = np.linalg.solve(matrix, -values[:, 0])
        barycentric = np.asarray([1.0 - uv.sum(), uv[0], uv[1]])
        if np.min(barycentric) < -1.0e-10 or np.max(barycentric) > 1.0 + 1.0e-10:
            continue
        point = mesh.p[:, triangle] @ barycentric
        if np.max(np.abs(point)) > search_config.search_half_extent_m:
            continue
        if not bool(recovered_field.geometry.contains_points(point)[0]):
            continue
        seeds.append(CandidateSeed(point, ("recovered-cell-zero",), 1, True))
    return tuple(seeds)


def _raw_element_seeds(
    recovered_field: RecoveredField,
    potential_v: ArrayLike,
    search_config: MinimaSearchConfig,
    limit: int,
) -> tuple[CandidateSeed, ...]:
    mesh = recovered_field.mesh
    raw = element_electric_fields(mesh, potential_v)
    psi = np.einsum("ij,ij->j", raw, raw)
    neighbors: list[list[int]] = [[] for _ in range(mesh.t.shape[1])]
    for left, right in mesh.f2t.T:
        if left >= 0 and right >= 0:
            neighbors[int(left)].append(int(right))
            neighbors[int(right)].append(int(left))
    centroids = np.mean(mesh.p[:, mesh.t], axis=1).T
    local = []
    for index, adjacent in enumerate(neighbors):
        if not adjacent or np.max(np.abs(centroids[index])) > search_config.search_half_extent_m:
            continue
        if psi[index] <= min(psi[item] for item in adjacent):
            local.append(index)
    ordered = sorted(local, key=lambda index: (float(psi[index]), index))[:limit]
    return tuple(
        CandidateSeed(
            centroids[index].copy(),
            ("raw-element-local-low",),
            1,
            True,
        )
        for index in ordered
    )


def _merge_seed_sources(
    seeds: Sequence[CandidateSeed],
    recovered_field: RecoveredField,
    tolerance_m: float,
) -> tuple[CandidateSeed, ...]:
    valued = sorted(
        (
            (float(recovered_field.pseudopotential(seed.position_m)), seed)
            for seed in seeds
        ),
        key=lambda item: item[0],
    )
    kept: list[CandidateSeed] = []
    for _, seed in valued:
        match = next(
            (
                index
                for index, item in enumerate(kept)
                if np.linalg.norm(seed.position_m - item.position_m) <= tolerance_m
            ),
            None,
        )
        if match is None:
            kept.append(seed)
            continue
        previous = kept[match]
        kept[match] = CandidateSeed(
            position_m=previous.position_m,
            source_names=tuple(sorted(set(previous.source_names + seed.source_names))),
            source_support_count=previous.source_support_count + seed.source_support_count,
            optimizer_succeeded=(
                previous.optimizer_succeeded or seed.optimizer_succeeded
            ),
        )
    return tuple(kept)


def _quality_table(
    seeds: Sequence[CandidateSeed],
    recovered_field: RecoveredField,
    potential_v: ArrayLike,
    search_config: MinimaSearchConfig,
    controls: RobustMinimaConfig,
    *,
    accept_by_robust_rules: bool,
) -> tuple[CandidateQualityMetrics, ...]:
    if not seeds:
        return ()
    psi_values = np.asarray(
        [float(recovered_field.pseudopotential(item.position_m)) for item in seeds]
    )
    ordered = np.sort(psi_values)
    reference_count = min(search_config.expected_minima, ordered.size)
    candidate_scale = max(
        float(np.median(ordered[:reference_count])),
        np.finfo(float).eps * max(float(np.max(ordered)), 1.0),
    )
    return tuple(
        compute_candidate_quality(
            candidate_id=index,
            seed=seed,
            recovered_field=recovered_field,
            potential_v=potential_v,
            candidate_psi_scale=candidate_scale,
            controls=controls,
            accept_by_robust_rules=accept_by_robust_rules,
        )
        for index, seed in enumerate(seeds, start=1)
    )


def compute_candidate_quality(
    *,
    candidate_id: int,
    seed: CandidateSeed,
    recovered_field: RecoveredField,
    potential_v: ArrayLike,
    candidate_psi_scale: float,
    controls: RobustMinimaConfig,
    accept_by_robust_rules: bool = True,
) -> CandidateQualityMetrics:
    """Compute all documented robust-quality metrics for one candidate."""

    if candidate_id <= 0:
        raise ValueError("candidate_id must be positive")
    if not np.isfinite(candidate_psi_scale) or candidate_psi_scale <= 0.0:
        raise ValueError("candidate_psi_scale must be finite and positive")
    mesh = recovered_field.mesh
    position = np.asarray(seed.position_m, dtype=float)
    recovered_psi = float(recovered_field.pseudopotential(position))
    raw_fields = element_electric_fields(mesh, potential_v)
    finder = mesh.element_finder()
    triangle = int(finder(position[0:1], position[1:2])[0])
    containing_magnitude = float(np.linalg.norm(raw_fields[:, triangle]))
    facet_distance, facet_index = nearest_internal_mesh_facet(mesh, position)
    adjacent = mesh.f2t[:, facet_index]
    adjacent_fields = raw_fields[:, adjacent]
    adjacent_magnitudes = tuple(
        float(np.linalg.norm(adjacent_fields[:, index])) for index in range(2)
    )
    jump = float(np.linalg.norm(adjacent_fields[:, 0] - adjacent_fields[:, 1]))
    jump_ratio = jump / max(*adjacent_magnitudes, np.finfo(float).tiny)
    mesh_h = float(mesh.param())
    steps = tuple(mesh_h * item for item in controls.hessian_step_mesh_fractions)
    eigenvalues = []
    for step in steps:
        hessian = finite_difference_hessian(recovered_field, position, step)
        if hessian is None:
            eigenvalues.append((float("nan"), float("nan")))
        else:
            values = np.linalg.eigvalsh(hessian)
            eigenvalues.append((float(values[0]), float(values[1])))
    stability, hessian_stable, variation = classify_hessian_stability(
        eigenvalues,
        maximum_variation_ratio=(
            controls.maximum_hessian_eigenvalue_variation_ratio
        ),
    )
    facet_fraction = facet_distance / mesh_h
    facet_sensitive = classify_facet_sensitive(
        facet_fraction,
        jump_ratio,
        hessian_stable,
        distance_threshold=controls.internal_facet_distance_mesh_fraction,
        jump_threshold=controls.adjacent_raw_field_jump_ratio,
    )
    psi_ratio = recovered_psi / max(candidate_psi_scale, np.finfo(float).tiny)
    small_psi = bool(psi_ratio <= controls.maximum_recovered_psi_ratio)
    legacy_facet_flag = bool(
        facet_fraction <= controls.internal_facet_distance_mesh_fraction
        and jump_ratio >= controls.adjacent_raw_field_jump_ratio
    )
    interpolation_sensitive = facet_sensitive or not hessian_stable or not small_psi
    reasons = []
    if not hessian_stable:
        reasons.append(stability)
    if facet_sensitive:
        reasons.append("facet-sensitive-unstable-signature")
    if not small_psi:
        reasons.append("high-recovered-psi-relative-to-candidate-set")
    robust_accepted = not reasons if accept_by_robust_rules else True
    if not reasons:
        reasons.append("accepted-stable-low-psi")
    probability = min(
        1.0,
        (0.25 if legacy_facet_flag else 0.0)
        + (0.20 if jump_ratio >= controls.adjacent_raw_field_jump_ratio else 0.0)
        + (0.40 if not hessian_stable else 0.0)
        + (0.15 if not small_psi else 0.0),
    )
    classification = "high" if probability >= 0.60 else "medium" if probability >= 0.30 else "low"
    return CandidateQualityMetrics(
        candidate_id=candidate_id,
        source_names=seed.source_names,
        source_support_count=seed.source_support_count,
        position_m=position.copy(),
        recovered_psi_v2_per_m2=recovered_psi,
        recovered_field_magnitude_v_per_m=float(np.sqrt(recovered_psi)),
        containing_raw_field_magnitude_v_per_m=containing_magnitude,
        adjacent_raw_field_magnitudes_v_per_m=adjacent_magnitudes,
        adjacent_raw_field_jump_v_per_m=jump,
        adjacent_raw_field_jump_ratio=jump_ratio,
        distance_to_internal_facet_m=facet_distance,
        distance_to_internal_facet_over_mesh_h=facet_fraction,
        distance_to_nearest_electrode_m=_electrode_clearance(recovered_field, position),
        hessian_steps_m=steps,
        hessian_eigenvalues_v2_per_m4=tuple(eigenvalues),
        hessian_stability_classification=stability,
        hessian_stable=hessian_stable,
        hessian_eigenvalue_variation_ratio=variation,
        recovered_psi_ratio_to_candidate_scale=psi_ratio,
        physically_small_recovered_psi=small_psi,
        facet_sensitive=facet_sensitive,
        legacy_interpolation_flag=legacy_facet_flag or not small_psi,
        interpolation_sensitive=interpolation_sensitive,
        artifact_probability=probability,
        artifact_classification=classification,
        classification_reason=";".join(reasons),
        robust_accepted=robust_accepted,
    )


def _electrode_clearance(
    recovered_field: RecoveredField,
    position_m: NDArray[np.float64],
) -> float:
    distances = (
        np.linalg.norm(recovered_field.geometry.centers_m - position_m, axis=1)
        - recovered_field.geometry.config.electrode_radius_m
    )
    return float(np.min(distances))


def _mark_selected(
    candidates: Sequence[CandidateQualityMetrics],
    minima: Sequence[LocalMinimum],
) -> tuple[CandidateQualityMetrics, ...]:
    tolerance = 1.0e-12
    return tuple(
        replace(
            item,
            selected=any(
                np.linalg.norm(item.position_m - minimum.position_m) <= tolerance
                for minimum in minima
            ),
        )
        for item in candidates
    )
