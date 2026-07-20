"""Coarse-to-fine local-minimum detection for the recovered field."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import minimum_filter
from scipy.optimize import minimize

from .config import MinimaSearchConfig
from .field import PointOutsideVacuumError, RecoveredField


class MinimaSearchError(RuntimeError):
    """Raised when the configured number of validated minima is not found."""


@dataclass(frozen=True)
class LocalMinimum:
    """One refined and Hessian-validated local pseudopotential minimum."""

    position_m: NDArray[np.float64]
    pseudopotential_v2_per_m2: float
    hessian_eigenvalues_v2_per_m4: NDArray[np.float64]
    optimizer_succeeded: bool

    @property
    def polar_angle_rad(self) -> float:
        """Return the position angle in the half-open interval ``[0, 2*pi)``."""

        angle = float(np.arctan2(self.position_m[1], self.position_m[0]))
        return angle % (2.0 * np.pi)


@dataclass(frozen=True)
class RefinedCandidate:
    """One unique recovered-field candidate before Hessian acceptance."""

    position_m: NDArray[np.float64]
    pseudopotential_v2_per_m2: float
    optimizer_succeeded: bool


@dataclass(frozen=True)
class RecoveredCandidateCollection:
    """All intermediate products of the legacy recovered-gradient search."""

    valid_coarse_points: int
    coarse_candidates: int
    refined_candidates: int
    unique_candidates: tuple[RefinedCandidate, ...]
    hessian_validated_minima: tuple[LocalMinimum, ...]


@dataclass(frozen=True)
class MinimaDiagnostics:
    """Counts and pre-selection results from the minima-search filters."""

    valid_coarse_points: int
    coarse_candidates: int
    refined_candidates: int
    unique_candidates: int
    hessian_validated_candidates: int
    hessian_validated_minima: tuple[LocalMinimum, ...] = ()


def find_local_minima(
    recovered_field: RecoveredField,
    config: MinimaSearchConfig,
) -> tuple[tuple[LocalMinimum, ...], MinimaDiagnostics]:
    """Find, refine, merge, validate, select, and angle-sort local minima."""

    collection = collect_recovered_candidates(recovered_field, config)
    validated = list(collection.hessian_validated_minima)
    diagnostics = MinimaDiagnostics(
        valid_coarse_points=collection.valid_coarse_points,
        coarse_candidates=collection.coarse_candidates,
        refined_candidates=collection.refined_candidates,
        unique_candidates=len(collection.unique_candidates),
        hessian_validated_candidates=len(validated),
        hessian_validated_minima=tuple(
            sorted(validated, key=lambda item: item.pseudopotential_v2_per_m2)
        ),
    )
    if len(validated) < config.expected_minima:
        raise MinimaSearchError(
            f"found {len(validated)} validated minima; expected {config.expected_minima}; "
            f"diagnostics={diagnostics}"
        )
    selected = sorted(
        validated,
        key=lambda item: item.pseudopotential_v2_per_m2,
    )[: config.expected_minima]
    return tuple(sorted(selected, key=lambda item: item.polar_angle_rad)), diagnostics


def collect_recovered_candidates(
    recovered_field: RecoveredField,
    config: MinimaSearchConfig,
) -> RecoveredCandidateCollection:
    """Run the legacy search while preserving candidates rejected by Hessian tests."""

    coarse_points, coarse_values, valid_count = _coarse_scan(recovered_field, config)
    candidate_points = _detect_candidates(coarse_points, coarse_values, config)
    refined = [
        _refine_candidate(recovered_field, point, config)
        for point in candidate_points
    ]
    refined = [item for item in refined if item is not None]
    unique_tuples = _merge_candidates(refined, config.merge_distance_m)
    unique = tuple(
        RefinedCandidate(
            position_m=item[0],
            pseudopotential_v2_per_m2=item[1],
            optimizer_succeeded=item[2],
        )
        for item in unique_tuples
    )
    validated = [
        minimum
        for minimum in (
            _validate_hessian(recovered_field, item, config) for item in unique
        )
        if minimum is not None
    ]
    return RecoveredCandidateCollection(
        valid_coarse_points=valid_count,
        coarse_candidates=len(candidate_points),
        refined_candidates=len(refined),
        unique_candidates=unique,
        hessian_validated_minima=tuple(
            sorted(validated, key=lambda item: item.pseudopotential_v2_per_m2)
        ),
    )


def refine_recovered_candidate(
    recovered_field: RecoveredField,
    start_m: NDArray[np.float64],
    config: MinimaSearchConfig,
) -> RefinedCandidate | None:
    """Refine one seed with the legacy recovered-field optimizer."""

    result = _refine_candidate(recovered_field, start_m, config)
    if result is None:
        return None
    return RefinedCandidate(
        position_m=result[0],
        pseudopotential_v2_per_m2=result[1],
        optimizer_succeeded=result[2],
    )


def finite_difference_hessian(
    recovered_field: RecoveredField,
    point_m: NDArray[np.float64],
    step_m: float,
) -> NDArray[np.float64] | None:
    """Evaluate the documented central-difference Hessian stencil."""

    return _finite_difference_hessian(recovered_field, point_m, step_m)


def _coarse_scan(
    recovered_field: RecoveredField,
    config: MinimaSearchConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    coordinates = np.linspace(
        -config.search_half_extent_m,
        config.search_half_extent_m,
        config.coarse_grid_points_per_axis,
    )
    grid_x, grid_y = np.meshgrid(coordinates, coordinates, indexing="xy")
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    valid = recovered_field.geometry.contains_points(points)
    values = np.full(points.shape[0], np.inf, dtype=float)
    if np.any(valid):
        values[valid] = np.asarray(recovered_field.pseudopotential(points[valid]))
    return points, values.reshape(grid_x.shape), int(np.count_nonzero(valid))


def _detect_candidates(
    coarse_points: NDArray[np.float64],
    coarse_values: NDArray[np.float64],
    config: MinimaSearchConfig,
) -> NDArray[np.float64]:
    filtered = minimum_filter(
        coarse_values,
        size=config.candidate_neighborhood,
        mode="constant",
        cval=np.inf,
    )
    mask = np.isfinite(coarse_values) & (coarse_values <= filtered)
    indices = np.flatnonzero(mask.ravel())
    if indices.size == 0:
        raise MinimaSearchError("coarse scan produced no finite local-minimum candidates")
    order = np.argsort(coarse_values.ravel()[indices], kind="stable")
    chosen = indices[order[: config.maximum_candidates]]
    return coarse_points[chosen]


def _refine_candidate(
    recovered_field: RecoveredField,
    start_m: NDArray[np.float64],
    config: MinimaSearchConfig,
) -> tuple[NDArray[np.float64], float, bool] | None:
    extent = config.search_half_extent_m
    initial_value = float(recovered_field.pseudopotential(start_m))
    objective_scale = max(initial_value, 1.0)

    def scaled_objective(scaled_position: NDArray[np.float64]) -> float:
        point_m = np.asarray(scaled_position, dtype=float) * extent
        if not bool(recovered_field.geometry.contains_points(point_m)[0]):
            return config.invalid_objective_v2_per_m2 / objective_scale
        try:
            return float(recovered_field.pseudopotential(point_m)) / objective_scale
        except PointOutsideVacuumError:
            return config.invalid_objective_v2_per_m2 / objective_scale

    result = minimize(
        scaled_objective,
        np.asarray(start_m, dtype=float) / extent,
        method="L-BFGS-B",
        bounds=((-1.0, 1.0), (-1.0, 1.0)),
        options={
            "eps": config.optimizer_step_m / extent,
            "ftol": config.optimizer_ftol,
            "maxiter": config.optimizer_max_iterations,
            "maxls": config.optimizer_max_line_search_steps,
        },
    )
    point_m = np.asarray(result.x, dtype=float) * extent
    if not bool(recovered_field.geometry.contains_points(point_m)[0]):
        return None
    try:
        value = float(recovered_field.pseudopotential(point_m))
    except PointOutsideVacuumError:
        return None
    return point_m, value, bool(result.success)


def _merge_candidates(
    candidates: list[tuple[NDArray[np.float64], float, bool]],
    tolerance_m: float,
) -> list[tuple[NDArray[np.float64], float, bool]]:
    unique: list[tuple[NDArray[np.float64], float, bool]] = []
    for candidate in sorted(candidates, key=lambda item: item[1]):
        if all(np.linalg.norm(candidate[0] - kept[0]) > tolerance_m for kept in unique):
            unique.append(candidate)
    return unique


def _validate_hessian(
    recovered_field: RecoveredField,
    candidate: RefinedCandidate,
    config: MinimaSearchConfig,
) -> LocalMinimum | None:
    position = candidate.position_m
    value = candidate.pseudopotential_v2_per_m2
    optimizer_succeeded = candidate.optimizer_succeeded
    hessian = _finite_difference_hessian(
        recovered_field,
        position,
        config.hessian_step_m,
    )
    if hessian is None:
        return None
    eigenvalues = np.linalg.eigvalsh(hessian)
    if np.any(eigenvalues <= config.minimum_hessian_eigenvalue_v2_per_m4):
        return None
    return LocalMinimum(
        position_m=position,
        pseudopotential_v2_per_m2=value,
        hessian_eigenvalues_v2_per_m4=eigenvalues,
        optimizer_succeeded=optimizer_succeeded,
    )


def _finite_difference_hessian(
    recovered_field: RecoveredField,
    point_m: NDArray[np.float64],
    step_m: float,
) -> NDArray[np.float64] | None:
    x, y = point_m
    stencil = np.asarray(
        [
            [x, y],
            [x + step_m, y],
            [x - step_m, y],
            [x, y + step_m],
            [x, y - step_m],
            [x + step_m, y + step_m],
            [x + step_m, y - step_m],
            [x - step_m, y + step_m],
            [x - step_m, y - step_m],
        ]
    )
    if not np.all(recovered_field.geometry.contains_points(stencil)):
        return None
    try:
        values = np.asarray(recovered_field.pseudopotential(stencil), dtype=float)
    except PointOutsideVacuumError:
        return None
    center, x_plus, x_minus, y_plus, y_minus, pp, pm, mp, mm = values
    h2 = step_m**2
    dxx = (x_plus - 2.0 * center + x_minus) / h2
    dyy = (y_plus - 2.0 * center + y_minus) / h2
    dxy = (pp - pm - mp + mm) / (4.0 * h2)
    return np.asarray([[dxx, dxy], [dxy, dyy]], dtype=float)
