"""Test functions for rainflow cycle counting.

Reference results come from the service load-time history of Lee (2025), Chapter 2,
Fig. 3A (points A..I):
    - Non-periodic history: the counting gives one closed cycle E-F and the residue
      A, B, C, D, G, H, I (Lee, Table 4 and Fig. 6), whose ranges are half cycles.
    - Periodic history: the rearranged history gives the full cycles E-F, A-B, H-C,
      and D-G (Lee, Table 3 and Fig. 4).

LEE, Yung-Li. Rainflow cycle counting methods. In: Metal Fatigue Testing and
Analysis, Chapter 2. Elsevier, 2025. https://doi.org/10.1016/B978-0-443-26665-2.00008-7

Tests cover:
    1. Three-point counting of Lee's history, its mirror, and the issue #62 example
       for non-periodic and periodic histories
    2. Half cycles from the starting-point rule and from the residue
    3. Periodic histories: full cycles only, cycles across the block end, rotation of
       the block, equivalence with a non-periodic history starting at its extreme
    4. Invariants on random signals (count balance, total variation, index validity,
       invariance to offset, scaling, and mirroring)
    5. Output shapes and dtypes
    6. Input validation (ValueError for wrong shape, empty, or non-finite signals)
"""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from fatpy.core.decompositions.uniaxial.rainflow import (
    calc_rainflow_cycles_three_point,
)
from fatpy.core.decompositions.uniaxial.reversals import find_reversals

# Lee (2025), Chapter 2, Fig. 3A service load-time history, points A..I
LEE_SIGNAL = np.array([2.0, -1.0, 3.0, -5.0, 1.0, -3.0, 4.0, -4.0, 2.0])

# Non-periodic: A-B, B-C (half), E-F (full), C-D, D-G, G-H, H-I (half)
LEE_CYCLE_INDICES = np.array([[0, 1], [1, 2], [4, 5], [2, 3], [3, 6], [6, 7], [7, 8]])
LEE_CYCLE_COUNTS = np.array([0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5])

# Periodic (Lee, Table 3): E-F, A-B, H-C, D-G (full)
LEE_PERIODIC_CYCLE_INDICES = np.array([[4, 5], [0, 1], [7, 2], [3, 6]])
LEE_PERIODIC_CYCLE_COUNTS = np.array([1.0, 1.0, 1.0, 1.0])

BASQUIN_SLOPE = 5.0


@pytest.fixture
def random_signals() -> list[NDArray[np.float64]]:
    """Fixture providing random signals, including repeated values and plateaus.

    Returns:
        list[NDArray[np.float64]]: Random signals of various lengths.
    """
    rng = np.random.default_rng(62)
    signals = [rng.normal(size=int(rng.integers(2, 300))) for _ in range(100)]
    # Rounding produces equal non-adjacent values and plateaus
    signals += [
        np.round(rng.normal(size=int(rng.integers(2, 300))), 1) for _ in range(100)
    ]
    return signals


def _relative_damage(
    signal: NDArray[np.float64],
    cycle_indices: NDArray[np.int64],
    cycle_counts: NDArray[np.float64],
) -> float:
    """Relative Palmgren-Miner damage sum(n * S_a^k) of the counted cycles."""
    values = signal[cycle_indices]
    amplitudes = np.abs(values[:, 1] - values[:, 0]) / 2.0
    return float(np.sum(cycle_counts * amplitudes**BASQUIN_SLOPE))


def _counted_variation(
    signal: NDArray[np.float64],
    cycle_indices: NDArray[np.int64],
    cycle_counts: NDArray[np.float64],
) -> float:
    """Total load variation represented by the counted cycles (2 * n * range)."""
    values = signal[cycle_indices]
    ranges = np.abs(values[:, 1] - values[:, 0])
    return float(np.sum(2.0 * cycle_counts * ranges))


# ---------------------------------------------------------------------------
# Three-point counting: reference histories
# ---------------------------------------------------------------------------


def test_three_point_lee_example() -> None:
    cycle_indices, cycle_counts = calc_rainflow_cycles_three_point(LEE_SIGNAL)

    np.testing.assert_array_equal(cycle_indices, LEE_CYCLE_INDICES)
    np.testing.assert_array_equal(cycle_counts, LEE_CYCLE_COUNTS)


def test_three_point_lee_closed_cycle_and_residue() -> None:
    """Closed cycle E-F and residue A, B, C, D, G, H, I agree with Lee, Table 4."""
    cycle_indices, cycle_counts = calc_rainflow_cycles_three_point(LEE_SIGNAL)

    full = cycle_indices[cycle_counts == 1.0]
    np.testing.assert_array_equal(full, [[4, 5]])  # E-F
    np.testing.assert_array_equal(LEE_SIGNAL[full[0]], [1.0, -3.0])  # range 4, mean -1

    # Residue ranges form a chain of half cycles through A, B, C, D, G, H, I
    half = cycle_indices[cycle_counts == 0.5]
    residue = np.append(half[:, 0], half[-1, 1])
    np.testing.assert_array_equal(residue, [0, 1, 2, 3, 6, 7, 8])
    np.testing.assert_array_equal(half[1:, 0], half[:-1, 1])


def test_three_point_lee_example_periodic() -> None:
    """Cycles E-F, A-B, H-C, D-G of the periodic history agree with Lee, Table 3."""
    cycle_indices, cycle_counts = calc_rainflow_cycles_three_point(
        LEE_SIGNAL, periodic=True
    )

    np.testing.assert_array_equal(cycle_indices, LEE_PERIODIC_CYCLE_INDICES)
    np.testing.assert_array_equal(cycle_counts, LEE_PERIODIC_CYCLE_COUNTS)
    np.testing.assert_array_equal(
        LEE_SIGNAL[cycle_indices],
        [[1.0, -3.0], [2.0, -1.0], [-4.0, 3.0], [-5.0, 4.0]],
    )


@pytest.mark.parametrize(
    "periodic, expected_indices, expected_counts",
    [
        (False, LEE_CYCLE_INDICES, LEE_CYCLE_COUNTS),
        (True, LEE_PERIODIC_CYCLE_INDICES, LEE_PERIODIC_CYCLE_COUNTS),
    ],
)
def test_three_point_mirrored_lee_example(
    periodic: bool,
    expected_indices: NDArray[np.int64],
    expected_counts: NDArray[np.float64],
) -> None:
    """The mirrored history (multiplied by -1) gives the same cycles."""
    cycle_indices, cycle_counts = calc_rainflow_cycles_three_point(
        -LEE_SIGNAL, periodic=periodic
    )

    np.testing.assert_array_equal(cycle_indices, expected_indices)
    np.testing.assert_array_equal(cycle_counts, expected_counts)


def test_three_point_issue_example() -> None:
    """Issue #62 example: S = [0, 2, 1, 3, 0] closes the cycle (1, 2)."""
    signal = [0.0, 2.0, 1.0, 3.0, 0.0]
    cycle_indices, cycle_counts = calc_rainflow_cycles_three_point(signal)

    np.testing.assert_array_equal(cycle_indices, [[1, 2], [0, 3], [3, 4]])
    np.testing.assert_array_equal(cycle_counts, [1.0, 0.5, 0.5])
    np.testing.assert_array_equal(cycle_indices[cycle_counts == 1.0], [[1, 2]])


def test_three_point_issue_example_periodic() -> None:
    """Repeated, the range 0-3 of the issue #62 example closes as well."""
    signal = [0.0, 2.0, 1.0, 3.0, 0.0]
    cycle_indices, cycle_counts = calc_rainflow_cycles_three_point(
        signal, periodic=True
    )

    np.testing.assert_array_equal(cycle_indices, [[1, 2], [0, 3]])
    np.testing.assert_array_equal(cycle_counts, [1.0, 1.0])


# ---------------------------------------------------------------------------
# Three-point counting: half cycles and special cases
# ---------------------------------------------------------------------------


def test_three_point_equal_ranges_containing_start_are_half_cycles() -> None:
    """With X = Y and Y containing the starting point, only half cycles are counted."""
    signal = [0.0, 1.0, 0.0, 1.0]
    cycle_indices, cycle_counts = calc_rainflow_cycles_three_point(signal)

    np.testing.assert_array_equal(cycle_indices, [[0, 1], [1, 2], [2, 3]])
    np.testing.assert_array_equal(cycle_counts, [0.5, 0.5, 0.5])


def test_three_point_equal_ranges_away_from_start_are_full_cycles() -> None:
    """X = Y closes a full cycle when Y does not contain the starting point."""
    # Points 1, 3, 1 (indices 2, 3, 4) give X = Y = 2 away from the start
    signal = [0.0, 5.0, 1.0, 3.0, 1.0, 4.0, -5.0]
    cycle_indices, cycle_counts = calc_rainflow_cycles_three_point(signal)

    np.testing.assert_array_equal(cycle_indices, [[2, 3], [4, 5], [0, 1], [1, 6]])
    np.testing.assert_array_equal(cycle_counts, [1.0, 1.0, 0.5, 0.5])


def test_three_point_indices_point_to_plateau_start() -> None:
    signal = [0.0, 2.0, 2.0, 1.0, 1.0, 3.0, 0.0]
    cycle_indices, cycle_counts = calc_rainflow_cycles_three_point(signal)

    np.testing.assert_array_equal(cycle_indices, [[1, 3], [0, 5], [5, 6]])
    np.testing.assert_array_equal(cycle_counts, [1.0, 0.5, 0.5])


@pytest.mark.parametrize(
    "signal, periodic, expected_indices, expected_counts",
    [
        ([3.0], False, np.empty((0, 2)), []),  # single sample
        ([3.0], True, np.empty((0, 2)), []),
        ([3.0, 3.0, 3.0], False, np.empty((0, 2)), []),  # constant signal
        ([3.0, 3.0, 3.0], True, np.empty((0, 2)), []),
        ([1.0, 4.0], False, [[0, 1]], [0.5]),  # single range
        ([1.0, 4.0], True, [[0, 1]], [1.0]),  # repeated: 1, 4, 1, 4, ...
        ([0.0, 1.0, 2.0, 3.0], False, [[0, 3]], [0.5]),  # monotonic signal
        ([0.0, 1.0, 2.0, 3.0], True, [[0, 3]], [1.0]),
    ],
)
def test_three_point_short_signals(
    signal: list[float],
    periodic: bool,
    expected_indices: NDArray[np.int64] | list[list[int]],
    expected_counts: list[float],
) -> None:
    cycle_indices, cycle_counts = calc_rainflow_cycles_three_point(
        signal, periodic=periodic
    )

    assert cycle_indices.shape == (len(expected_counts), 2)
    np.testing.assert_array_equal(cycle_indices, expected_indices)
    np.testing.assert_array_equal(cycle_counts, expected_counts)


# ---------------------------------------------------------------------------
# Three-point counting: periodic histories
# ---------------------------------------------------------------------------


def test_three_point_periodic_last_sample_equal_to_first_is_same_point() -> None:
    """Appending the first sample to the end of a block does not change the cycles."""
    block = [0.0, 5.0, -5.0]
    open_indices, open_counts = calc_rainflow_cycles_three_point(block, periodic=True)
    closed_indices, closed_counts = calc_rainflow_cycles_three_point(
        [*block, 0.0], periodic=True
    )

    np.testing.assert_array_equal(open_indices, [[1, 2]])
    np.testing.assert_array_equal(closed_indices, open_indices)
    np.testing.assert_array_equal(closed_counts, open_counts)


def test_three_point_periodic_only_full_cycles(
    random_signals: list[NDArray[np.float64]],
) -> None:
    for signal in random_signals:
        cycle_indices, cycle_counts = calc_rainflow_cycles_three_point(
            signal, periodic=True
        )

        assert np.all(cycle_counts == 1.0)
        assert np.all(np.isin(cycle_indices, find_reversals(np.r_[signal, signal])))
        assert np.all((cycle_indices >= 0) & (cycle_indices < signal.size))


def test_three_point_periodic_invariant_to_block_rotation(
    random_signals: list[NDArray[np.float64]],
) -> None:
    """Any block of the same repeated history gives the same cycles."""
    rng = np.random.default_rng(0)
    for signal in random_signals:
        shift = int(rng.integers(0, signal.size))
        rotated = np.roll(signal, shift)

        expected = calc_rainflow_cycles_three_point(signal, periodic=True)
        result = calc_rainflow_cycles_three_point(rotated, periodic=True)

        expected_values = np.sort(signal[expected[0]], axis=1)
        result_values = np.sort(rotated[result[0]], axis=1)
        np.testing.assert_array_equal(
            result_values[np.lexsort(result_values.T)],
            expected_values[np.lexsort(expected_values.T)],
        )


def test_three_point_history_starting_at_extreme_periodic_equivalence(
    random_signals: list[NDArray[np.float64]],
) -> None:
    """A history starting and ending at its maximum gives equal damage in both modes."""
    for random_signal in random_signals:
        peak = random_signal.max() + 1.0
        signal = np.r_[peak, random_signal, peak]

        non_periodic = calc_rainflow_cycles_three_point(signal)
        periodic = calc_rainflow_cycles_three_point(signal, periodic=True)

        assert _relative_damage(signal, *non_periodic) == pytest.approx(
            _relative_damage(signal, *periodic), rel=1e-12
        )


# ---------------------------------------------------------------------------
# Three-point counting: invariants
# ---------------------------------------------------------------------------


def test_three_point_count_balance(random_signals: list[NDArray[np.float64]]) -> None:
    """Each range between reversals is used once: 2 * N_full + N_half = m."""
    for signal in random_signals:
        _, cycle_counts = calc_rainflow_cycles_three_point(signal)
        n_ranges = len(find_reversals(signal)) - 1

        n_full = np.count_nonzero(cycle_counts == 1.0)
        n_half = np.count_nonzero(cycle_counts == 0.5)

        assert n_full + n_half == len(cycle_counts)
        assert 2 * n_full + n_half == n_ranges


@pytest.mark.parametrize("periodic", [False, True])
def test_three_point_total_variation(
    random_signals: list[NDArray[np.float64]], periodic: bool
) -> None:
    """The counted cycles represent the whole up and down movement of the signal."""
    for signal in random_signals:
        cycle_indices, cycle_counts = calc_rainflow_cycles_three_point(
            signal, periodic=periodic
        )
        # A periodic block also includes the return from its end to its start
        path = np.r_[signal, signal[0]] if periodic else signal
        expected = float(np.sum(np.abs(np.diff(path))))

        assert _counted_variation(signal, cycle_indices, cycle_counts) == (
            pytest.approx(expected, rel=1e-12, abs=1e-12)
        )


def test_three_point_indices_are_reversals(
    random_signals: list[NDArray[np.float64]],
) -> None:
    for signal in random_signals:
        cycle_indices, _ = calc_rainflow_cycles_three_point(signal)
        reversals = find_reversals(signal)

        assert np.all(np.isin(cycle_indices, reversals))
        assert np.all(cycle_indices[:, 0] < cycle_indices[:, 1])
        assert np.all(signal[cycle_indices[:, 0]] != signal[cycle_indices[:, 1]])


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize(
    "transform",
    [
        lambda s: s + 100.0,
        lambda s: s - 250.0,
        lambda s: 3.5 * s,
        lambda s: -s,
    ],
    ids=["offset_up", "offset_down", "scale", "mirror"],
)
def test_three_point_invariant_to_offset_scale_mirror(
    random_signals: list[NDArray[np.float64]],
    transform: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    periodic: bool,
) -> None:
    for random_signal in random_signals:
        # Integer-valued samples keep the transformed ranges exact, so equal
        # ranges (X = Y) stay equal and the comparison is not a rounding artefact
        signal = np.round(10.0 * random_signal)
        expected_indices, expected_counts = calc_rainflow_cycles_three_point(
            signal, periodic=periodic
        )
        cycle_indices, cycle_counts = calc_rainflow_cycles_three_point(
            transform(signal), periodic=periodic
        )

        np.testing.assert_array_equal(cycle_indices, expected_indices)
        np.testing.assert_array_equal(cycle_counts, expected_counts)


# ---------------------------------------------------------------------------
# Output types and input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("periodic", [False, True])
def test_three_point_output_types(periodic: bool) -> None:
    cycle_indices, cycle_counts = calc_rainflow_cycles_three_point(
        LEE_SIGNAL, periodic=periodic
    )

    assert cycle_indices.dtype == np.int64
    assert cycle_counts.dtype == np.float64
    assert cycle_indices.shape == (len(cycle_counts), 2)


def test_three_point_accepts_integer_list() -> None:
    signal = [2, -1, 3, -5, 1, -3, 4, -4, 2]
    cycle_indices, cycle_counts = calc_rainflow_cycles_three_point(signal)

    np.testing.assert_array_equal(cycle_indices, LEE_CYCLE_INDICES)
    np.testing.assert_array_equal(cycle_counts, LEE_CYCLE_COUNTS)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize(
    "signal, message",
    [
        (np.zeros((3, 2)), "one-dimensional"),
        (np.float64(1.0), "one-dimensional"),
        (np.array([]), "at least one sample"),
        (np.array([0.0, np.nan, 1.0]), "NaN or infinite"),
        (np.array([0.0, np.inf, 1.0]), "NaN or infinite"),
    ],
)
def test_three_point_invalid_signal_raises(
    signal: NDArray[np.float64], message: str, periodic: bool
) -> None:
    with pytest.raises(ValueError, match=message):
        calc_rainflow_cycles_three_point(signal, periodic=periodic)
