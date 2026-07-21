"""Unit tests for Milestone 9 refinement and calibration logic."""

from __future__ import annotations

import numpy as np
import pytest

from rf_trap_forward.calibrated_validation import (
    CalibrationCase,
    CalibrationEvaluation,
    CalibrationRow,
    GeometryVariant,
    OutputTransform,
    VoltageModel,
    apply_output_transform,
    default_geometry_variant,
    fit_output_transform,
    generate_geometry_variants,
    normalize_voltage_vector,
    passes_validation_gate,
    rank_calibration_evaluations,
)
from rf_trap_forward.reference_validation import MinimumMatch, match_minima_by_distance


def test_geometry_parameter_variants_span_requested_ranges() -> None:
    """The curated scan must cover center, electrode, outer, and interpretation axes."""

    variants = generate_geometry_variants()
    centers_mm = {round(1.0e3 * item.electrode_center_radius_m, 6) for item in variants}
    electrode_mm = {round(1.0e3 * item.electrode_radius_m, 6) for item in variants}
    outer_mm = {round(1.0e3 * item.outer_boundary_radius_m, 6) for item in variants}
    assert {18.0, 20.0, 21.48, 23.0, 25.0} <= centers_mm
    assert {8.0, 9.0, 10.0, 11.0, 12.0} <= electrode_mm
    assert {30.0, 40.0, 50.0, 65.0, 80.0} <= outer_mm
    assert any("diagonal-coordinate" in item.interpretation for item in variants)
    with pytest.raises(ValueError, match="overlapping"):
        GeometryVariant("invalid", 15.0e-3, 12.0e-3, 50.0e-3)


def test_voltage_normalization_and_global_gauge_shift() -> None:
    """Normalization must bound voltages and shift every boundary consistently."""

    raw = np.asarray((1.0, 0.8, -0.5, 0.2))
    normalized, outer = normalize_voltage_vector(raw)
    shifted, shifted_outer = normalize_voltage_vector(
        raw,
        remove_global_offset=True,
    )
    assert max(abs(value - outer) for value in normalized) == pytest.approx(1.0)
    assert max(abs(value - shifted_outer) for value in shifted) == pytest.approx(1.0)
    np.testing.assert_allclose(
        np.asarray(normalized) - outer,
        np.asarray(shifted) - shifted_outer,
    )
    with pytest.raises(ValueError, match="constant"):
        normalize_voltage_vector((1.0, 1.0, 1.0, 1.0), outer_potential_v=1.0)


def test_output_transform_fit_recovers_synthetic_calibration() -> None:
    """The bounded diagnostic fit should reconstruct a known transformed cloud."""

    raw = np.asarray(((-0.003, 0.001), (0.002, 0.003), (0.001, -0.004)))
    expected = OutputTransform(1.12, 7.0, 1.08)
    target = apply_output_transform(raw, expected)
    fitted = fit_output_transform((raw,), (target,))
    reconstructed = apply_output_transform(raw, fitted)
    matches = np.asarray(
        [item.distance_m for item in match_minima_by_distance(target, reconstructed)]
    )
    assert np.max(matches) < 2.0e-6


def test_calibration_ranking_prioritizes_complete_low_error_results() -> None:
    """Ranking must expose the best Data.txt fit while the gate audits topology."""

    complete = _evaluation("complete", (0.20e-3, 0.30e-3, 0.25e-3), topology=3)
    lower_error_bad_topology = _evaluation(
        "bad-topology",
        (0.01e-3, 0.01e-3, 0.01e-3),
        topology=4,
    )
    ranked = rank_calibration_evaluations((lower_error_bad_topology, complete))
    assert ranked[0].case.name == "bad-topology"
    assert not ranked[0].summary().validation_gate_passed


def test_validation_gate_requires_error_and_topology_limits() -> None:
    """The conservative gate must require all rows and both error thresholds."""

    assert passes_validation_gate(
        selected_rows=10,
        completed_rows=10,
        exactly_three_rows=10,
        mean_error_m=0.25e-3,
        maximum_error_m=0.50e-3,
    )
    assert not passes_validation_gate(
        selected_rows=10,
        completed_rows=10,
        exactly_three_rows=9,
        mean_error_m=0.10e-3,
        maximum_error_m=0.20e-3,
    )
    assert not passes_validation_gate(
        selected_rows=10,
        completed_rows=10,
        exactly_three_rows=10,
        mean_error_m=0.26e-3,
        maximum_error_m=0.40e-3,
    )


def _evaluation(
    name: str,
    errors: tuple[float, float, float],
    *,
    topology: int,
) -> CalibrationEvaluation:
    reference = np.asarray(((-1.0e-3, 0.0), (0.0, 1.0e-3), (1.0e-3, 0.0)))
    matches = tuple(
        MinimumMatch(
            reference_index=index + 1,
            computed_index=index + 1,
            reference_position_m=point.copy(),
            computed_position_m=point + np.asarray((error, 0.0)),
            delta_m=np.asarray((error, 0.0)),
            distance_m=error,
        )
        for index, (point, error) in enumerate(zip(reference, errors, strict=True))
    )
    computed = np.vstack([item.computed_position_m for item in matches])
    row = CalibrationRow(
        hypothesis_name=name,
        family="mock",
        scope="mock",
        row_number=1,
        status="ok",
        reference_positions_m=reference,
        raw_computed_positions_m=computed,
        computed_positions_m=computed,
        matches=matches,
        topology_candidate_count=topology,
        selected_interpolation_sensitive=0,
        rejected_candidates=0,
        total_candidates=topology,
        node_count=10,
        triangle_count=12,
        relative_free_residual=1.0e-15,
        runtime_seconds=0.1,
    )
    case = CalibrationCase(
        name=name,
        family="mock",
        geometry=default_geometry_variant(),
        voltage=VoltageModel("all-positive", (1.0, 1.0, 1.0, 1.0)),
    )
    return CalibrationEvaluation(case, "mock", 0.5e-3, (row,))
