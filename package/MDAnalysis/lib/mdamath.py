# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# MDAnalysis --- https://www.mdanalysis.org
# Copyright (c) 2006-2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
#
# Please cite your use of MDAnalysis in published work:
#
# R. J. Gowers, M. Linke, J. Barnoud, T. J. E. Reddy, M. N. Melo, S. L. Seyler,
# D. L. Dotson, J. Domanski, S. Buchoux, I. M. Kenney, and O. Beckstein.
# MDAnalysis: A Python package for the rapid analysis of molecular dynamics
# simulations. In S. Benthall and S. Rostrup editors, Proceedings of the 15th
# Python in Science Conference, pages 102-109, Austin, TX, 2016. SciPy.
# doi: 10.25080/majora-629e541a-00e
#
# N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, and O. Beckstein.
# MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics Simulations.
# J. Comput. Chem. 32 (2011), 2319--2327, doi:10.1002/jcc.21787
#

"""
Mathematical helper functions --- :mod:`MDAnalysis.lib.mdamath`
===============================================================

Helper functions for common mathematical operations

.. autofunction:: normal
.. autofunction:: norm
.. autofunction:: angle
.. autofunction:: dihedral
.. autofunction:: stp
.. autofunction:: sarrus_det
.. autofunction:: triclinic_box
.. autofunction:: triclinic_vectors
.. autofunction:: box_volume
.. autofunction:: make_whole
.. autofunction:: find_fragments

.. versionadded:: 0.11.0
"""
from __future__ import division, absolute_import
from six.moves import zip
import numpy as np

from ..exceptions import NoDataError
from . import util
from ._cutil import (make_whole, find_fragments, _sarrus_det_single,
                     _sarrus_det_multiple)

# geometric functions
def norm(v):
    r"""Calculate the norm of a vector v.

    .. math:: v = \sqrt{\mathbf{v}\cdot\mathbf{v}}

    This version is faster then numpy.linalg.norm because it only works for a
    single vector and therefore can skip a lot of the additional fuss
    linalg.norm does.

    Parameters
    ----------
    v : array_like
        1D array of shape (N) for a vector of length N

    Returns
    -------
    float
        norm of the vector

    """
    return np.sqrt(np.dot(v, v))


def normal(vec1, vec2):
    r"""Returns the unit vector normal to two vectors.

    .. math::

       \hat{\mathbf{n}} = \frac{\mathbf{v}_1 \times \mathbf{v}_2}{|\mathbf{v}_1 \times \mathbf{v}_2|}

    If the two vectors are collinear, the vector :math:`\mathbf{0}` is returned.

    .. versionchanged:: 0.11.0
       Moved into lib.mdamath
    """
    normal = np.cross(vec1, vec2)
    n = norm(normal)
    if n == 0.0:
        return normal  # returns [0,0,0] instead of [nan,nan,nan]
    return normal / n  # ... could also use numpy.nan_to_num(normal/norm(normal))


def angle(a, b):
    """Returns the angle between two vectors in radians

    .. versionchanged:: 0.11.0
       Moved into lib.mdamath
    """
    x = np.dot(a, b) / (norm(a) * norm(b))
    # catch roundoffs that lead to nan otherwise
    x = np.clip(x, -1.0, 1.0)
    return np.arccos(x)


def stp(vec1, vec2, vec3):
    r"""Takes the scalar triple product of three vectors.

    Returns the volume *V* of the parallel epiped spanned by the three
    vectors

    .. math::

        V = \mathbf{v}_3 \cdot (\mathbf{v}_1 \times \mathbf{v}_2)

    .. versionchanged:: 0.11.0
       Moved into lib.mdamath
    """
    return np.dot(vec3, np.cross(vec1, vec2))


def dihedral(ab, bc, cd):
    r"""Returns the dihedral angle in radians between vectors connecting A,B,C,D.

    The dihedral measures the rotation around bc::

         ab
       A---->B
              \ bc
              _\'
                C---->D
                  cd

    The dihedral angle is restricted to the range -π <= x <= π.

    .. versionadded:: 0.8
    .. versionchanged:: 0.11.0
       Moved into lib.mdamath
    """
    x = angle(normal(ab, bc), normal(bc, cd))
    return (x if stp(ab, bc, cd) <= 0.0 else -x)


def _angle(a, b):
    """Angle between two vectors *a* and *b* in degrees.

    If one of the lengths is 0 then the angle is returned as 0
    (instead of `nan`).
    """
    # This function has different limits than angle?

    angle = np.arccos(np.dot(a, b) / (norm(a) * norm(b)))
    if np.isnan(angle):
        return 0.0
    return np.rad2deg(angle)


def sarrus_det(matrix):
    """Computes the determinant of a 3x3 matrix according to the
    `rule of Sarrus`_.

    If an array of 3x3 matrices is supplied, determinants are computed per
    matrix and returned as an appropriately shaped numpy array.

    .. _rule of Sarrus:
       https://en.wikipedia.org/wiki/Rule_of_Sarrus

    Parameters
    ----------
    matrix : numpy.ndarray
        An array of shape ``(..., 3, 3)`` with the 3x3 matrices residing in the
        last two dimensions.

    Returns
    -------
    det : float or numpy.ndarray
        The determinant(s) of `matrix`.
        If ``matrix.shape == (3, 3)``, the determinant will be returned as a
        scalar. If ``matrix.shape == (..., 3, 3)``, the determinants will be
        returned as a :class:`numpy.ndarray` of shape ``(...,)`` and dtype
        ``numpy.float64``.

    Raises
    ------
    ValueError:
        If `matrix` has less than two dimensions or its last two dimensions
        are not of shape ``(3, 3)``
