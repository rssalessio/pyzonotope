from __future__ import annotations
import numpy as np
import cvxpy as cp
from typing import Union, Tuple
from copy import deepcopy
from scipy.linalg import block_diag
from pyzonotope.interval_matrix import IntervalMatrix
from pyzonotope.zonotope import Zonotope
from pyzonotope.interval import Interval

def to_cvxpy(x: Union[list,np.ndarray,cp.Expression]) -> Union[cp.Expression,cp.Constant]:
    assert isinstance(x, list) or isinstance(x, np.ndarray) \
        or isinstance(x, cp.Expression), 'Value is neither a list, an array or a cvxpy expression'
    
    if isinstance(x, list) or isinstance(x, np.ndarray):
        return cp.Constant(x)

    return x

class CVXZonotope(object):
    """
    CVX Zonotope - object constructor for zonotope objects
    See also CORA library (https://github.com/TUMcps/CORA/blob/master/contSet/%40zonotope/zonotope.m)

    Offers reduced functionalities compared to the Zonotope class
    
    Description:
        This class represents zonotopes objects defined as
        {c + \sum_{i=1}^p beta_i * g^(i) | beta_i \in [-1,1]}.
    
    """
    Z: cp.Expression


    def __init__(self, center: Union[list,np.ndarray,cp.Expression], generator: Union[list,np.ndarray,cp.Expression]):
        center = to_cvxpy(center)
        generator = to_cvxpy(generator)

        assert center.shape[0] == generator.shape[0], 'Center and generator do not have the same number of rows'
        assert len(generator.shape) == 2, 'Generator must be a matrix'
        assert len(center.shape) == 1 or \
            len(center.shape) == 2 and center.shape[1] == 1, \
            'Center must be a column vector'

        self.Z = cp.hstack([
            center if len(center.shape) == 2 else center[:, np.newaxis],
            generator
        ])
    
    @property
    def center(self) -> cp.Expression:
        """ Returns the center of the Zonotope of dimension n"""
        return self.Z[:, :1].flatten()

    @property
    def generators(self) -> cp.Expression:
        """
        Returns the generators of the zonotope (matrix of dimension n x g),
        where n is the dimensionality and g is the number of generators.
        """
        return self.Z[:, 1:]

    @property
    def num_generators(self) -> int:
        """ Number of generators """
        return self.generators.shape[1]

    @property
    def dimension(self) -> int:
        """ Dimensionality """
        return self.generators.shape[0]

    @property
    def shape(self) -> Tuple[int,int]:
        """ Returns d x n, where d is the dimensionality and n is the number of generators """
        return [self.dimension, self.num_generators]

    def copy(self) -> CVXZonotope:
        """ Returns a copy of the zonotope """
        return CVXZonotope(deepcopy(self.center)[:, np.newaxis], deepcopy(self.generators))

    def __add__(self, operand: Union[float, int, np.ndarray, Zonotope, CVXZonotope]) -> CVXZonotope:
        if isinstance(operand, float) or isinstance(operand, int) or isinstance(operand, np.ndarray) or isinstance(operand, cp.Expression):
            return CVXZonotope(self.Z[:, 0] + operand, self.Z[:, 1:])
        if isinstance(operand, Zonotope) or isinstance(operand, CVXZonotope):
            assert np.all(operand.dimension == self.dimension), \
                f"Operand has not the same dimension, {self.dimension} != {operand.dimension}"
            return CVXZonotope(self.Z[:, 0] + operand.center, cp.hstack([self.Z[:,1:], operand.generators]))
        else:
            raise Exception(f"Addition not implemented for type {type(operand)}")

    def __mul__(self, operand: Union[int, float, np.ndarray, IntervalMatrix]) -> CVXZonotope:
        if isinstance(operand, float) or isinstance(operand, int):
            Z = self.Z * operand
            return CVXZonotope(Z[:,0], Z[:, 1:])
        elif isinstance(operand, np.ndarray):
            # Left multiplication, operand * self
            if operand.shape[1] == self.center.shape[0]:
                Z = operand @ self.Z
                return CVXZonotope(Z[:,0], Z[:, 1:])
            # Right multiplication, self * operand
            # It's the same as the left multiplication
            elif operand.shape[0] == self.center.shape[1]:
                return self.__mul__(operand.T)

            else:
                raise Exception('Incorrect dimension')

        elif isinstance(operand, IntervalMatrix):
            T = operand.center
            S = operand.radius
            Zabssum = cp.sum(cp.abs(self.Z), axis=1)
            Z = cp.hstack([T @ self.Z, cp.diag(S @ Zabssum)])
            return CVXZonotope(Z[:,0], Z[:, 1:])

        else:
            raise Exception(f"Multiplication not implemented for type {type(operand)}")


    __rmul__ = __mul__   # commutative operation
    __matmul__ = __mul__

    def __str__(self):
        return f'Center: {self.center.value} - Generators: {self.generators.value}'

    @property
    def interval(self) -> Interval:
        """ Return the interval representation of the zonotope """
        center = self.center
        delta = cp.sum(cp.abs(self.generators), axis=1)
        return Interval(center - delta, center + delta)

    def cartesian_product(self, W: Union[Zonotope, CVXZonotope]) -> CVXZonotope:
        """
        Returns the cartesian product ZxW between Z and W

        :param W: Zonotope object
        :return: Zonotope object
        """
        assert isinstance(W, Zonotope) or isinstance(W, CVXZonotope), 'Operand is not a Zonotope'
        new_center = cp.hstack((self.center, W.center))
        A = np.zeros((self.generators.shape[0], W.generators.shape[1]))
        B = np.zeros((W.generators.shape[0], self.generators.shape[1]))
        new_generators = cp.bmat([[self.generators, A], [B, W.generators]])
        return CVXZonotope(new_center, new_generators)

    def over_approximate(self) -> CVXZonotope:
        """
        Over approximates a zonotope by a cube
        """
        delta = cp.sum(cp.abs(self.generators), axis=1)[:, None]
        return CVXZonotope(self.center, delta)