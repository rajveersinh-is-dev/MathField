"""
mathfield/core/basis/

Stage 2: Global Discovery Basis Function Library.
"""

from mathfield.core.basis.base import BasisFunction2D
from mathfield.core.basis.fourier import DCT2Basis, Fourier2D
from mathfield.core.basis.polynomial import Chebyshev2D, Legendre2D
from mathfield.core.basis.rational import Rational2D
from mathfield.core.basis.rbf import (
    GaussianRBF,
    MultiquadricRBF,
    WendlandC2RBF,
    generate_halton_centers,
    generate_residual_peak_centers,
    generate_uniform_centers,
)
from mathfield.core.basis.splines import BSplineSurface2D

__all__ = [
    "BasisFunction2D",
    "Chebyshev2D",
    "Legendre2D",
    "Fourier2D",
    "DCT2Basis",
    "WendlandC2RBF",
    "GaussianRBF",
    "MultiquadricRBF",
    "generate_uniform_centers",
    "generate_halton_centers",
    "generate_residual_peak_centers",
    "BSplineSurface2D",
    "Rational2D",
]
