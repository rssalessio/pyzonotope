from __future__ import annotations
import numpy as np
import cvxpy as cp
from typing import List, Union, Tuple
from pyzonotope.zonotope import Zonotope
from copy import deepcopy
from pyzonotope.interval_matrix import IntervalMatrix
from pyzonotope.cvx_zonotope import CVXZonotope


class MatrixZonotope(object):
    """
    MatZonotope class.
    
    Represents the set of matrices of dimension (n,p) contained in the zonotope
    
    See also CORA Library
    https://github.com/TUMcps/CORA/blob/master/matrixSet/%40matZonotope/matZonotope.m
    """
    dim_n: int
    dim_p: int
    num_generators: int
    
    """
    Matrix of dimension (g+1,n,p), where g is the number of generators.
    Z[0] contains the center, and Z[1:] contains the generators
    """
    Z: np.ndarray


    def __init__(self, center: np.ndarray, generators: np.ndarray):
        assert len(center.shape) == 2 and len(generators.shape) == 3, \
            "Center must be a matrix and generators a tensor"
        assert center.shape == generators.shape[1:], \
            "Center and generators must have the same dimensions"


        self.Z = np.zeros((generators.shape[0] + 1, center.shape[0], center.shape[1]))
        self.Z[0] = center
        self.Z[1:] = generators

        self.dim_n = center.shape[0]
        self.dim_p = center.shape[1]
        self.num_generators = generators.shape[0]

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.dim_n, self.dim_p)

    @property
    def center(self) -> np.ndarray:
        """ Returns the center of the matrix zonotope, of dimension (n,p)"""
        return self.Z[0]
    
    @property
    def generators(self) -> np.ndarray:
        """ Returns the generators of the matrix zonotope, of dimension (g,n,p) where g is the number of generators """
        return self.Z[1:]
    
    @property
    def order(self) -> float:
        """ Returns the order of the matrix zonotope.
            Equivalent to converting the matrix zonotope to a zonotope and then computing the order of this zonotope
        """
        return self.num_generators / np.prod(self.shape)

    def copy(self) -> MatrixZonotope:
        return MatrixZonotope(deepcopy(self.center), deepcopy(self.generators))

    def __add__(self, operand: Union[float, int, np.ndarray, MatrixZonotope]) -> MatrixZonotope:
        if(isinstance(operand, np.ndarray)) or isinstance(operand, int) or isinstance(operand, float):
            if isinstance(operand, np.ndarray):
                assert operand.shape == self.center.shape, 'Incorrect shape for operand'
            return MatrixZonotope(self.center + operand, self.generators)
        elif isinstance(operand, MatrixZonotope):
            assert self.shape == operand.shape, 'Incorrect dimensionality for operand'
            return MatrixZonotope(self.center + operand.center, np.concatenate((self.generators, operand.generators), axis=0))
        else:
            raise NotImplementedError(f'Add not implemented for {type(operand)}')

    __radd__ = __add__

    def __mul__(self, operand: Union[int, float, np.ndarray, Zonotope, CVXZonotope]) -> Union[MatrixZonotope, CVXZonotope, Zonotope]:
        if isinstance(operand, float) or isinstance(operand, int):
            return MatrixZonotope(self.center * operand, self.generators * operand)
        
        elif isinstance(operand, Zonotope):
            assert self.center.shape[1] == operand.Z.shape[0]
            # This is equivalent to np.concatenate(np.matmul(self.Z, operand.Z[None]), axis=1)
            # but it's much faster to reshape
            Znew = np.matmul(self.Z, operand.Z[None]).swapaxes(0,1).reshape(self.dim_n, -1)
            return Zonotope(Znew[:,:1], Znew[:, 1:])

        elif isinstance(operand, CVXZonotope):
            assert self.center.shape[1] == operand.Z.shape[0]
            Znew: List[cp.Expression] = [self.center @ operand.Z]
        
            for i in range(self.num_generators):
                Zadd: cp.Expression = self.generators[i] @ operand.Z
                Znew.append(Zadd)

            Znew = cp.hstack(Znew)        
            return CVXZonotope(Znew[:,:1], Znew[:, 1:])

        elif isinstance(operand, np.ndarray):
            # Right multiplication, self * operand
            if operand.shape[0] == self.center.shape[1]:
                # @TODO Add matmul implementation
                center = self.center @ operand
                generators = np.zeros((self.num_generators, self.dim_n, operand.shape[1]))
                
                for i in range(self.num_generators):
                    generators[i, :, :] = self.generators[i] @ operand
            # Left multiplication, operand * self
            elif operand.shape[1] == self.center.shape[0]:
                # @TODO Add matmul implementation
                center = operand @ self.center
                generators = np.zeros((self.num_generators, operand.shape[0], self.dim_p))
                
                for i in range(self.num_generators):
                    generators[i, :, :] = operand @ self.generators[i]
            else:
                raise Exception('Invalid dimension')
            
            return MatrixZonotope(center, generators)
        else:
            raise NotImplementedError

    __rmul__ = __mul__

    def __str__(self):
        return f'Center: {self.center} - Generators: {self.generators.T}'

    def sample(self, batch_size: int = 1) -> np.ndarray:
        """
        Generates a uniform random points within a matrix zonotope

        Return a tensor of size (b x n x p), where (n,p) is the dimensionality
        of a single sample, and b is the number of samples

        :param batch_size: number of random points
        :return: A tensor where each element is a sample
        """
        beta = np.random.uniform(low=-1, high=1, size=(batch_size, self.num_generators))
        return (self.center[None, :] + np.tensordot(beta, self.generators, axes=1))

    @property
    def interval_matrix(self) -> IntervalMatrix:

        delta = np.abs(self.generators[0])
        for i in range(1, self.num_generators):
            delta += np.abs(self.generators[i])

        return IntervalMatrix(self.center, delta)

    def contains(self, X: np.ndarray, tolerance: float = 1e-9) -> bool:
        """
        Returns true if the matrix zonotope contains X
        """
        return self.interval_matrix.contains(X, tolerance)

    def over_approximate(self) -> MatrixZonotope:
        """
        Over approximates a Matrix zonotope by a cube
        """
        delta = np.abs(self.generators[0])
        for i in range(1, self.num_generators):
            delta += np.abs(self.generators[i])
        return MatrixZonotope(self.center, delta[None,:])

    @property
    def zonotope(self) -> Zonotope:
        """ Convert the matrix zonotope to a zonotope """
        center = self.center.flatten()
        generators = [self.generators[i].flatten()[:, None] for i in range(self.num_generators)]
        return Zonotope(center, np.hstack(generators))
    
    def reduce(self, order: int) -> MatrixZonotope:
        """
        Reduces the matrix zonotope using Zonotopes reduction methods

        :param order: desired order
        :return: a Matrix zonotope
        """
        # Convert to zonotope and reduce
        reduced_zonotope = self.zonotope.reduce(order)

        # Convert zonotope back to matrix zonotope
        center = reduced_zonotope.center.reshape(self.center.shape)
        generators = np.zeros((reduced_zonotope.num_generators, center.shape[0], center.shape[1]))

        for i in range(reduced_zonotope.num_generators):
            generators[i] = reduced_zonotope.generators[:,i].reshape(center.shape)

        # Return new matrix zonotope
        return MatrixZonotope(center, generators)

    @property
    def max_norm(self) -> Tuple[float, np.ndarray]:
        """ Returns the maximum infinity norm on the set """
        return self.zonotope.max_norm

    def compute_vertices(self):
        """
        Uses convex hull algorithm
        """
        vertices= self.zonotope.compute_vertices()

        V = np.zeros((vertices.shape[0], self.dim_n, self.dim_p))
        # Convert zonotope back to matrix zonotope
        for i in range(vertices.shape[0]):
            V[i] = vertices[i].reshape(self.center.shape)

        return V

    def choose_columns(self, idxs: np.ndarray) -> MatrixZonotope:
        return MatrixZonotope(self.center[:, idxs], self.generators[:, :, idxs])

    def choose_rows(self, idxs: np.ndarray) -> MatrixZonotope:
        return MatrixZonotope(self.center[idxs, :], self.generators[:, idxs, :])

