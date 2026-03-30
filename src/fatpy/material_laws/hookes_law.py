"""Hooke's law.

Converts between elastic stress and strain for linear elastic, isotropic materials.

Provides both 3D and plane stress formulations. Each formulation includes functions
for assembling compliance and stiffness matrices, as well as direct stress-to-strain
and strain-to-stress conversions.

Conventions:
    - Isotropic material behavior is assumed throughout this module.
    - 3D vectors use Voigt notation with shape (..., 6), where the last dimension
      contains the six Voigt components:

          (σ_11, σ_22, σ_33, σ_23, σ_13, σ_12)

    - Plane stress vectors use reduced Voigt notation with shape (..., 3):

          (σ_11, σ_22, σ_12)

    - Shear strain components are tensor shear strains (ε_ij), not engineering
      shear strains (γ_ij = 2ε_ij), consistent with the rest of FatPy.
    - Leading dimensions are preserved in all functions, enabling batch processing.
"""

import numpy as np
from numpy.typing import NDArray

from fatpy.utils import voigt

PLANE_STRESS_COMPONENTS_COUNT = 3


def _check_plane_stress_shape(vector: NDArray[np.float64]) -> None:
    """Validate the plane stress vector shape.

    Args:
        vector: Array with shape (..., 3) where the last dimension has length 3.

    Raises:
        ValueError: If the last dimension is not of size 3.
    """
    if vector.shape[-1] != PLANE_STRESS_COMPONENTS_COUNT:
        raise ValueError(
            "Last dimension must correspond to 3 plane stress "
            "components (..., 3): (σ_11, σ_22, σ_12) or (ε_11, ε_22, ε_12)."
        )


# ---------------------------------------------------------------------------
# 3D Hooke's Law
# ---------------------------------------------------------------------------


def calc_compliance_matrix_3d(
    elastic_modulus: float,
    poisson_ratio: float,
) -> NDArray[np.float64]:
    r"""Assemble the 3D compliance matrix for an isotropic linear elastic material.

    Maps stress to strain: ε = S · σ.

    ??? abstract "Math Equations"
        The compliance matrix in tensor shear strain convention:

        $$
        \mathbf{S} = \frac{1}{E}
        \begin{bmatrix}
        1      & -\nu   & -\nu   & 0        & 0        & 0        \\
        -\nu   & 1      & -\nu   & 0        & 0        & 0        \\
        -\nu   & -\nu   & 1      & 0        & 0        & 0        \\
        0      & 0      & 0      & (1+\nu)  & 0        & 0        \\
        0      & 0      & 0      & 0        & (1+\nu)  & 0        \\
        0      & 0      & 0      & 0        & 0        & (1+\nu)
        \end{bmatrix}
        $$

        The shear diagonal entry $(1+\nu)/E = 1/(2G)$ corresponds to tensor
        shear strain $\varepsilon_{ij}$, not engineering shear strain
        $\gamma_{ij} = 2\varepsilon_{ij}$.

    Args:
        elastic_modulus: Young's modulus E [MPa]. Must be positive.
        poisson_ratio: Poisson's ratio ν [-]. Must satisfy -1 < ν < 0.5.

    Returns:
        Array of shape (6, 6). The compliance matrix.
    """
    e = elastic_modulus
    nu = poisson_ratio

    s = np.zeros((6, 6), dtype=np.float64)

    # Normal block
    diag = 1.0 / e
    off = -nu / e
    s[0, 0] = s[1, 1] = s[2, 2] = diag
    s[0, 1] = s[0, 2] = s[1, 0] = s[1, 2] = s[2, 0] = s[2, 1] = off

    # Shear block — tensor shear strain: 1/(2G) = (1+ν)/E
    s[3, 3] = s[4, 4] = s[5, 5] = (1.0 + nu) / e

    return s


def calc_stiffness_matrix_3d(
    elastic_modulus: float,
    poisson_ratio: float,
) -> NDArray[np.float64]:
    r"""Assemble the 3D stiffness matrix for an isotropic linear elastic material.

    Maps strain to stress: σ = C · ε.

    ??? abstract "Math Equations"
        Using Lamé parameters $\lambda$ and $\mu = G$, the stiffness matrix in
        tensor shear strain convention:

        $$
        \mathbf{C} =
        \begin{bmatrix}
        \lambda + 2\mu & \lambda        & \lambda        & 0     & 0     & 0     \\
        \lambda        & \lambda + 2\mu & \lambda        & 0     & 0     & 0     \\
        \lambda        & \lambda        & \lambda + 2\mu & 0     & 0     & 0     \\
        0              & 0              & 0              & 2\mu  & 0     & 0     \\
        0              & 0              & 0              & 0     & 2\mu  & 0     \\
        0              & 0              & 0              & 0     & 0     & 2\mu
        \end{bmatrix}
        $$

        where $\lambda = \frac{E\nu}{(1+\nu)(1-2\nu)}$ and
        $\mu = G = \frac{E}{2(1+\nu)}$.

        The shear diagonal entry $2\mu = 2G$ maps tensor shear strain directly
        to shear stress: $\sigma_{ij} = 2G \, \varepsilon_{ij}$.

    Args:
        elastic_modulus: Young's modulus E [MPa]. Must be positive.
        poisson_ratio: Poisson's ratio ν [-]. Must satisfy -1 < ν < 0.5.

    Returns:
        Array of shape (6, 6). The stiffness matrix.
    """
    e = elastic_modulus
    nu = poisson_ratio

    lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = e / (2.0 * (1.0 + nu))

    c = np.zeros((6, 6), dtype=np.float64)

    # Normal block
    c[0, 0] = c[1, 1] = c[2, 2] = lam + 2.0 * mu
    c[0, 1] = c[0, 2] = c[1, 0] = c[1, 2] = c[2, 0] = c[2, 1] = lam

    # Shear block — tensor shear strain: 2G = 2μ
    c[3, 3] = c[4, 4] = c[5, 5] = 2.0 * mu

    return c


def calc_strain_3d(
    elastic_modulus: float,
    poisson_ratio: float,
    stress_vector_voigt: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Convert 3D stress to strain using Hooke's law for isotropic material.

    ??? abstract "Math Equations"
        $$ \boldsymbol{\varepsilon} = \mathbf{S} \, \boldsymbol{\sigma} $$

        where $\mathbf{S}$ is the 3D compliance matrix.

    Args:
        elastic_modulus: Young's modulus E [MPa]. Must be positive.
        poisson_ratio: Poisson's ratio ν [-]. Must satisfy -1 < ν < 0.5.
        stress_vector_voigt: Array of shape (..., 6). Stress components in
            Voigt notation.

    Returns:
        Array of shape (..., 6). Strain components in Voigt notation
            (tensor shear strain convention).

    Raises:
        ValueError: If the last dimension is not of size 6.
    """
    voigt.check_shape(stress_vector_voigt)

    compliance = calc_compliance_matrix_3d(elastic_modulus, poisson_ratio)

    result: NDArray[np.float64] = np.einsum(
        "ij,...j->...i", compliance, stress_vector_voigt
    )
    return result


def calc_stress_3d(
    elastic_modulus: float,
    poisson_ratio: float,
    strain_vector_voigt: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Convert 3D strain to stress using Hooke's law for isotropic material.

    ??? abstract "Math Equations"
        $$ \boldsymbol{\sigma} = \mathbf{C} \, \boldsymbol{\varepsilon} $$

        where $\mathbf{C}$ is the 3D stiffness matrix.

    Args:
        elastic_modulus: Young's modulus E [MPa]. Must be positive.
        poisson_ratio: Poisson's ratio ν [-]. Must satisfy -1 < ν < 0.5.
        strain_vector_voigt: Array of shape (..., 6). Strain components in
            Voigt notation (tensor shear strain convention).

    Returns:
        Array of shape (..., 6). Stress components in Voigt notation.

    Raises:
        ValueError: If the last dimension is not of size 6.
    """
    voigt.check_shape(strain_vector_voigt)

    stiffness = calc_stiffness_matrix_3d(elastic_modulus, poisson_ratio)

    result: NDArray[np.float64] = np.einsum(
        "ij,...j->...i", stiffness, strain_vector_voigt
    )
    return result


# ---------------------------------------------------------------------------
# Plane Stress Hooke's Law
# ---------------------------------------------------------------------------


def calc_compliance_matrix_plane_stress(
    elastic_modulus: float,
    poisson_ratio: float,
) -> NDArray[np.float64]:
    r"""Assemble the plane stress compliance matrix for isotropic material.

    Maps in-plane stress to in-plane strain: ε = S · σ for the plane stress
    condition (σ_33 = σ_13 = σ_23 = 0).

    ??? abstract "Math Equations"
        The compliance matrix in tensor shear strain convention:

        $$
        \mathbf{S} = \frac{1}{E}
        \begin{bmatrix}
        1      & -\nu   & 0        \\
        -\nu   & 1      & 0        \\
        0      & 0      & (1+\nu)
        \end{bmatrix}
        $$

    Note:
        The out-of-plane strain $\varepsilon_{33} = -\frac{\nu}{E}
        (\sigma_{11} + \sigma_{22})$ is not included in the output.

    Args:
        elastic_modulus: Young's modulus E [MPa]. Must be positive.
        poisson_ratio: Poisson's ratio ν [-]. Must satisfy -1 < ν < 0.5.

    Returns:
        Array of shape (3, 3). The plane stress compliance matrix.
    """
    e = elastic_modulus
    nu = poisson_ratio

    return np.array(
        [
            [1.0 / e, -nu / e, 0.0],
            [-nu / e, 1.0 / e, 0.0],
            [0.0, 0.0, (1.0 + nu) / e],
        ],
        dtype=np.float64,
    )


def calc_stiffness_matrix_plane_stress(
    elastic_modulus: float,
    poisson_ratio: float,
) -> NDArray[np.float64]:
    r"""Assemble the plane stress stiffness matrix for isotropic material.

    Maps in-plane strain to in-plane stress: σ = C · ε for the plane stress
    condition.

    ??? abstract "Math Equations"
        The stiffness matrix in tensor shear strain convention:

        $$
        \mathbf{C} = \frac{E}{1-\nu^2}
        \begin{bmatrix}
        1    & \nu  & 0        \\
        \nu  & 1    & 0        \\
        0    & 0    & (1-\nu)
        \end{bmatrix}
        $$

        The shear entry $\frac{E(1-\nu)}{1-\nu^2} = \frac{E}{1+\nu} = 2G$
        maps tensor shear strain directly to shear stress.

    Args:
        elastic_modulus: Young's modulus E [MPa]. Must be positive.
        poisson_ratio: Poisson's ratio ν [-]. Must satisfy -1 < ν < 0.5.

    Returns:
        Array of shape (3, 3). The plane stress stiffness matrix.
    """
    e = elastic_modulus
    nu = poisson_ratio
    factor = e / (1.0 - nu**2)

    return factor * np.array(
        [
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, 1.0 - nu],
        ],
        dtype=np.float64,
    )


def calc_strain_plane_stress(
    elastic_modulus: float,
    poisson_ratio: float,
    stress_vector_voigt: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Convert in-plane stress to strain under plane stress using Hooke's law.

    Assumes isotropic linear elastic material with σ_33 = σ_13 = σ_23 = 0.

    ??? abstract "Math Equations"
        $$ \boldsymbol{\varepsilon} = \mathbf{S} \, \boldsymbol{\sigma} $$

        where $\mathbf{S}$ is the plane stress compliance matrix.

    Note:
        The out-of-plane strain $\varepsilon_{33} = -\frac{\nu}{E}
        (\sigma_{11} + \sigma_{22})$ is not included in the output.

    Args:
        elastic_modulus: Young's modulus E [MPa]. Must be positive.
        poisson_ratio: Poisson's ratio ν [-]. Must satisfy -1 < ν < 0.5.
        stress_vector_voigt: Array of shape (..., 3). Plane stress components
            in reduced Voigt notation: (σ_11, σ_22, σ_12).

    Returns:
        Array of shape (..., 3). In-plane strain in reduced Voigt notation:
            (ε_11, ε_22, ε_12), tensor shear strain convention.

    Raises:
        ValueError: If the last dimension is not of size 3.
    """
    _check_plane_stress_shape(stress_vector_voigt)

    compliance = calc_compliance_matrix_plane_stress(elastic_modulus, poisson_ratio)

    result: NDArray[np.float64] = np.einsum(
        "ij,...j->...i", compliance, stress_vector_voigt
    )
    return result


def calc_stress_plane_stress(
    elastic_modulus: float,
    poisson_ratio: float,
    strain_vector_voigt: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Convert in-plane strain to stress under plane stress using Hooke's law.

    Assumes isotropic linear elastic material.

    ??? abstract "Math Equations"
        $$ \boldsymbol{\sigma} = \mathbf{C} \, \boldsymbol{\varepsilon} $$

        where $\mathbf{C}$ is the plane stress stiffness matrix.

    Args:
        elastic_modulus: Young's modulus E [MPa]. Must be positive.
        poisson_ratio: Poisson's ratio ν [-]. Must satisfy -1 < ν < 0.5.
        strain_vector_voigt: Array of shape (..., 3). In-plane strain in reduced
            Voigt notation: (ε_11, ε_22, ε_12), tensor shear strain convention.

    Returns:
        Array of shape (..., 3). Plane stress in reduced Voigt notation:
            (σ_11, σ_22, σ_12).

    Raises:
        ValueError: If the last dimension is not of size 3.
    """
    _check_plane_stress_shape(strain_vector_voigt)

    stiffness = calc_stiffness_matrix_plane_stress(elastic_modulus, poisson_ratio)

    result: NDArray[np.float64] = np.einsum(
        "ij,...j->...i", stiffness, strain_vector_voigt
    )
    return result
