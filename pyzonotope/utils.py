import numpy as np
import cvxpy as cp
from pyzonotope.zonotope import Zonotope
from pyzonotope.matrix_zonotope import MatrixZonotope
from typing import Union


def concatenate_zonotope(zonotope: Zonotope, N: int) -> MatrixZonotope:
    """
    Concatenates a zonotope N times (creating a matrix zonotope)
    of dimension (N*g, n, N), where the first dimension is the number of generators,
    and g is the number of generators in the zonotope

    :param zonotope: Zonotope
    :param N: number of concatenations
    :return: Matrix Zonotope
    """
    assert N > 0 and isinstance(N, int), 'N must be a positive integer'
    dim_x = zonotope.dimension
    C = np.tile(zonotope.center, (N, 1)).T
    G = np.zeros((N * zonotope.num_generators, dim_x, N))

    for i in range(zonotope.num_generators):
        for j in range(N):
            G[j + i*N, :, j] = zonotope.generators[:, i]

    return MatrixZonotope(C, G)

