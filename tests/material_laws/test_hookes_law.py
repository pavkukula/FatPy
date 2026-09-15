"""Test functions for Hooke's law in material_laws.

Tests cover:
    1. Compliance and stiffness matrix properties (symmetry, mutual inverse)
    2. Stress-strain roundtrip consistency (σ → ε → σ and ε → σ → ε)
    3. Known analytical cases (uniaxial tension, pure shear, hydrostatic)
    4. Output shapes for 1D, 2D, and 3D input arrays
    5. Shape validation (ValueError for wrong last dimension)
    6. Plane stress specific behaviour
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from fatpy.material_laws import hookes_law


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

E_STEEL = 210_000.0  # MPa
NU_STEEL = 0.3


@pytest.fixture
def steel_params() -> tuple[float, float]:
    """Typical steel material parameters."""
    return E_STEEL, NU_STEEL


@pytest.fixture
def stress_3d_sample() -> NDArray[np.float64]:
    """Sample 3D stress states with shape (2, 3, 6)."""
    return np.array(
        [
            [
                [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Uniaxial x
                [0.0, 0.0, 0.0, 0.0, 0.0, 50.0],  # Pure shear xy
                [100.0, 100.0, 100.0, 0.0, 0.0, 0.0],  # Hydrostatic
            ],
            [
                [200.0, -100.0, 0.0, 0.0, 0.0, 0.0],  # Biaxial
                [0.0, 0.0, 0.0, 30.0, 0.0, 0.0],  # Pure shear yz
                [50.0, 50.0, 0.0, 0.0, 0.0, 25.0],  # Plane stress-like
            ],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def stress_ps_sample() -> NDArray[np.float64]:
    """Sample plane stress states with shape (2, 3, 3)."""
    return np.array(
        [
            [
                [100.0, 0.0, 0.0],  # Uniaxial x
                [0.0, 0.0, 50.0],  # Pure shear
                [100.0, 100.0, 0.0],  # Equibiaxial
            ],
            [
                [200.0, -100.0, 0.0],  # Biaxial
                [0.0, 200.0, 0.0],  # Uniaxial y
                [50.0, 50.0, 25.0],  # Mixed
            ],
        ],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# 3D — Matrix properties
# ---------------------------------------------------------------------------


def test_compliance_matrix_3d_symmetry(
    steel_params: tuple[float, float],
) -> None:
    """Compliance matrix must be symmetric."""
    s = hookes_law.calc_compliance_matrix_3d(*steel_params)
    assert np.allclose(s, s.T, atol=1e-15)


def test_stiffness_matrix_3d_symmetry(
    steel_params: tuple[float, float],
) -> None:
    """Stiffness matrix must be symmetric."""
    c = hookes_law.calc_stiffness_matrix_3d(*steel_params)
    assert np.allclose(c, c.T, atol=1e-15)


def test_compliance_stiffness_inverse_3d(
    steel_params: tuple[float, float],
) -> None:
    """Compliance and stiffness matrices must be mutual inverses."""
    s = hookes_law.calc_compliance_matrix_3d(*steel_params)
    c = hookes_law.calc_stiffness_matrix_3d(*steel_params)
    identity = np.eye(6)
    assert np.allclose(s @ c, identity, atol=1e-10)
    assert np.allclose(c @ s, identity, atol=1e-10)


# ---------------------------------------------------------------------------
# 3D — Roundtrip consistency
# ---------------------------------------------------------------------------


def test_stress_strain_roundtrip_3d(
    steel_params: tuple[float, float],
    stress_3d_sample: NDArray[np.float64],
) -> None:
    """σ → ε → σ must recover the original stress."""
    e_mod, nu = steel_params
    strain = hookes_law.calc_strain_3d(e_mod, nu, stress_3d_sample)
    stress_recovered = hookes_law.calc_stress_3d(e_mod, nu, strain)
    assert np.allclose(stress_recovered, stress_3d_sample, atol=1e-8)


def test_strain_stress_roundtrip_3d(
    steel_params: tuple[float, float],
) -> None:
    """ε → σ → ε must recover the original strain."""
    e_mod, nu = steel_params
    strain_input = np.array(
        [0.001, -0.0003, -0.0003, 0.0002, 0.0, 0.0005], dtype=np.float64
    )
    stress = hookes_law.calc_stress_3d(e_mod, nu, strain_input)
    strain_recovered = hookes_law.calc_strain_3d(e_mod, nu, stress)
    assert np.allclose(strain_recovered, strain_input, atol=1e-12)


# ---------------------------------------------------------------------------
# 3D — Known analytical cases
# ---------------------------------------------------------------------------


def test_uniaxial_tension_3d(
    steel_params: tuple[float, float],
) -> None:
    """Uniaxial tension σ_11 = σ must give ε_11 = σ/E, ε_22 = ε_33 = -ν·σ/E."""
    e_mod, nu = steel_params
    sigma = 100.0
    stress = np.array([sigma, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    strain = hookes_law.calc_strain_3d(e_mod, nu, stress)

    assert np.isclose(strain[0], sigma / e_mod, atol=1e-15)
    assert np.isclose(strain[1], -nu * sigma / e_mod, atol=1e-15)
    assert np.isclose(strain[2], -nu * sigma / e_mod, atol=1e-15)
    assert np.allclose(strain[3:], 0.0, atol=1e-15)


def test_pure_shear_3d(
    steel_params: tuple[float, float],
) -> None:
    """Pure shear σ_12 = τ must give ε_12 = τ(1+ν)/E = τ/(2G)."""
    e_mod, nu = steel_params
    tau = 50.0
    stress = np.array([0.0, 0.0, 0.0, 0.0, 0.0, tau], dtype=np.float64)

    strain = hookes_law.calc_strain_3d(e_mod, nu, stress)

    g = e_mod / (2.0 * (1.0 + nu))
    expected_tensor_shear = tau / (2.0 * g)
    assert np.isclose(strain[5], expected_tensor_shear, atol=1e-15)
    assert np.allclose(strain[:5], 0.0, atol=1e-15)


def test_hydrostatic_stress_3d(
    steel_params: tuple[float, float],
) -> None:
    """Hydrostatic stress p must give ε_ii = p(1-2ν)/E for each normal component."""
    e_mod, nu = steel_params
    p = 100.0
    stress = np.array([p, p, p, 0.0, 0.0, 0.0], dtype=np.float64)

    strain = hookes_law.calc_strain_3d(e_mod, nu, stress)

    expected_normal = p * (1.0 - 2.0 * nu) / e_mod
    assert np.allclose(strain[:3], expected_normal, atol=1e-15)
    assert np.allclose(strain[3:], 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# 3D — Output shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        (6,),
        (5, 6),
        (2, 3, 6),
        (4, 2, 3, 6),
    ],
)
def test_calc_strain_3d_shape(
    steel_params: tuple[float, float],
    shape: tuple[int, ...],
) -> None:
    """Output shape must match input shape for calc_strain_3d."""
    stress = np.ones(shape, dtype=np.float64)
    strain = hookes_law.calc_strain_3d(*steel_params, stress)
    assert strain.shape == shape


@pytest.mark.parametrize(
    "shape",
    [
        (6,),
        (5, 6),
        (2, 3, 6),
        (4, 2, 3, 6),
    ],
)
def test_calc_stress_3d_shape(
    steel_params: tuple[float, float],
    shape: tuple[int, ...],
) -> None:
    """Output shape must match input shape for calc_stress_3d."""
    strain = np.ones(shape, dtype=np.float64)
    stress = hookes_law.calc_stress_3d(*steel_params, strain)
    assert stress.shape == shape


# ---------------------------------------------------------------------------
# 3D — Shape validation
# ---------------------------------------------------------------------------


def test_calc_strain_3d_wrong_shape(
    steel_params: tuple[float, float],
) -> None:
    """Must raise ValueError for wrong last dimension."""
    with pytest.raises(ValueError):
        hookes_law.calc_strain_3d(*steel_params, np.ones((3,), dtype=np.float64))


def test_calc_stress_3d_wrong_shape(
    steel_params: tuple[float, float],
) -> None:
    """Must raise ValueError for wrong last dimension."""
    with pytest.raises(ValueError):
        hookes_law.calc_stress_3d(*steel_params, np.ones((4,), dtype=np.float64))


# ---------------------------------------------------------------------------
# Plane Stress — Matrix properties
# ---------------------------------------------------------------------------


def test_compliance_matrix_ps_symmetry(
    steel_params: tuple[float, float],
) -> None:
    """Plane stress compliance matrix must be symmetric."""
    s = hookes_law.calc_compliance_matrix_plane_stress(*steel_params)
    assert np.allclose(s, s.T, atol=1e-15)


def test_stiffness_matrix_ps_symmetry(
    steel_params: tuple[float, float],
) -> None:
    """Plane stress stiffness matrix must be symmetric."""
    c = hookes_law.calc_stiffness_matrix_plane_stress(*steel_params)
    assert np.allclose(c, c.T, atol=1e-15)


def test_compliance_stiffness_inverse_ps(
    steel_params: tuple[float, float],
) -> None:
    """Plane stress compliance and stiffness must be mutual inverses."""
    s = hookes_law.calc_compliance_matrix_plane_stress(*steel_params)
    c = hookes_law.calc_stiffness_matrix_plane_stress(*steel_params)
    identity = np.eye(3)
    assert np.allclose(s @ c, identity, atol=1e-10)
    assert np.allclose(c @ s, identity, atol=1e-10)


# ---------------------------------------------------------------------------
# Plane Stress — Roundtrip consistency
# ---------------------------------------------------------------------------


def test_stress_strain_roundtrip_ps(
    steel_params: tuple[float, float],
    stress_ps_sample: NDArray[np.float64],
) -> None:
    """σ → ε → σ must recover the original stress (plane stress)."""
    e_mod, nu = steel_params
    strain = hookes_law.calc_strain_plane_stress(e_mod, nu, stress_ps_sample)
    stress_recovered = hookes_law.calc_stress_plane_stress(e_mod, nu, strain)
    assert np.allclose(stress_recovered, stress_ps_sample, atol=1e-8)


def test_strain_stress_roundtrip_ps(
    steel_params: tuple[float, float],
) -> None:
    """ε → σ → ε must recover the original strain (plane stress)."""
    e_mod, nu = steel_params
    strain_input = np.array([0.001, -0.0003, 0.0005], dtype=np.float64)
    stress = hookes_law.calc_stress_plane_stress(e_mod, nu, strain_input)
    strain_recovered = hookes_law.calc_strain_plane_stress(e_mod, nu, stress)
    assert np.allclose(strain_recovered, strain_input, atol=1e-12)


# ---------------------------------------------------------------------------
# Plane Stress — Known analytical cases
# ---------------------------------------------------------------------------


def test_uniaxial_tension_ps(
    steel_params: tuple[float, float],
) -> None:
    """Uniaxial tension σ_11 = σ must give ε_11 = σ/E, ε_22 = -ν·σ/E."""
    e_mod, nu = steel_params
    sigma = 100.0
    stress = np.array([sigma, 0.0, 0.0], dtype=np.float64)

    strain = hookes_law.calc_strain_plane_stress(e_mod, nu, stress)

    assert np.isclose(strain[0], sigma / e_mod, atol=1e-15)
    assert np.isclose(strain[1], -nu * sigma / e_mod, atol=1e-15)
    assert np.isclose(strain[2], 0.0, atol=1e-15)


def test_pure_shear_ps(
    steel_params: tuple[float, float],
) -> None:
    """Pure shear σ_12 = τ must give ε_12 = τ/(2G) (tensor shear)."""
    e_mod, nu = steel_params
    tau = 50.0
    stress = np.array([0.0, 0.0, tau], dtype=np.float64)

    strain = hookes_law.calc_strain_plane_stress(e_mod, nu, stress)

    g = e_mod / (2.0 * (1.0 + nu))
    expected_tensor_shear = tau / (2.0 * g)
    assert np.isclose(strain[2], expected_tensor_shear, atol=1e-15)
    assert np.allclose(strain[:2], 0.0, atol=1e-15)


def test_equibiaxial_tension_ps(
    steel_params: tuple[float, float],
) -> None:
    """Equibiaxial σ_11 = σ_22 = σ must give ε_11 = ε_22 = σ(1-ν)/E."""
    e_mod, nu = steel_params
    sigma = 100.0
    stress = np.array([sigma, sigma, 0.0], dtype=np.float64)

    strain = hookes_law.calc_strain_plane_stress(e_mod, nu, stress)

    expected = sigma * (1.0 - nu) / e_mod
    assert np.isclose(strain[0], expected, atol=1e-15)
    assert np.isclose(strain[1], expected, atol=1e-15)
    assert np.isclose(strain[2], 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# Plane Stress — Consistency with 3D
# ---------------------------------------------------------------------------


def test_plane_stress_consistent_with_3d(
    steel_params: tuple[float, float],
) -> None:
    """Plane stress result must match 3D result when σ_33 = σ_13 = σ_23 = 0."""
    e_mod, nu = steel_params
    stress_ps = np.array([100.0, -50.0, 30.0], dtype=np.float64)
    stress_3d = np.array([100.0, -50.0, 0.0, 0.0, 0.0, 30.0], dtype=np.float64)

    strain_ps = hookes_law.calc_strain_plane_stress(e_mod, nu, stress_ps)
    strain_3d = hookes_law.calc_strain_3d(e_mod, nu, stress_3d)

    # In-plane components must agree
    assert np.isclose(strain_ps[0], strain_3d[0], atol=1e-12)
    assert np.isclose(strain_ps[1], strain_3d[1], atol=1e-12)
    assert np.isclose(strain_ps[2], strain_3d[5], atol=1e-12)  # ε_12


# ---------------------------------------------------------------------------
# Plane Stress — Output shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        (3,),
        (5, 3),
        (2, 3, 3),
        (4, 2, 3, 3),
    ],
)
def test_calc_strain_ps_shape(
    steel_params: tuple[float, float],
    shape: tuple[int, ...],
) -> None:
    """Output shape must match input shape for calc_strain_plane_stress."""
    stress = np.ones(shape, dtype=np.float64)
    strain = hookes_law.calc_strain_plane_stress(*steel_params, stress)
    assert strain.shape == shape


@pytest.mark.parametrize(
    "shape",
    [
        (3,),
        (5, 3),
        (2, 3, 3),
        (4, 2, 3, 3),
    ],
)
def test_calc_stress_ps_shape(
    steel_params: tuple[float, float],
    shape: tuple[int, ...],
) -> None:
    """Output shape must match input shape for calc_stress_plane_stress."""
    strain = np.ones(shape, dtype=np.float64)
    stress = hookes_law.calc_stress_plane_stress(*steel_params, strain)
    assert stress.shape == shape


# ---------------------------------------------------------------------------
# Plane Stress — Shape validation
# ---------------------------------------------------------------------------


def test_calc_strain_ps_wrong_shape(
    steel_params: tuple[float, float],
) -> None:
    """Must raise ValueError for wrong last dimension."""
    with pytest.raises(ValueError):
        hookes_law.calc_strain_plane_stress(
            *steel_params, np.ones((6,), dtype=np.float64)
        )


def test_calc_stress_ps_wrong_shape(
    steel_params: tuple[float, float],
) -> None:
    """Must raise ValueError for wrong last dimension."""
    with pytest.raises(ValueError):
        hookes_law.calc_stress_plane_stress(
            *steel_params, np.ones((5,), dtype=np.float64)
        )
