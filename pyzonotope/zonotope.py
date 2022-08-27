from __future__ import annotations
import numpy as np
import cvxpy as cp
from cmath import isclose
from typing import Union, Tuple
from copy import deepcopy
from scipy.linalg import block_diag
from scipy.special import comb
from scipy.spatial import ConvexHull
from pyzonotope.interval import Interval
from pyzonotope.interval_matrix import IntervalMatrix

class Zonotope(object):
    """
    zonotope - object constructor for zonotope objects
    See also CORA library (https://github.com/TUMcps/CORA/blob/master/contSet/%40zonotope/zonotope.m)
    
    Description:
        This class represents zonotopes objects defined as
        {c + \sum_{i=1}^p beta_i * g^(i) | beta_i \in [-1,1]}.
    
    """
    Z: np.ndarray
    half_space: np.ndarray

    def __init__(self, center: Union[list,np.ndarray], generator: Union[list,np.ndarray]):
        center = np.array(center)
        generator = np.array(generator)

        assert center.shape[0] == generator.shape[0], 'Center and generator do not have the same number of rows'
        assert len(generator.shape) == 2, 'Generator must be a matrix'
        assert len(center.shape) == 1 or \
            len(center.shape) == 2 and center.shape[1] == 1, \
            'Center must be a column vector'

        self.Z = np.hstack([
            center if len(center.shape) == 2 else center[:, np.newaxis],
            generator
        ])
        self.half_space = np.array([])
    
    @property
    def center(self) -> np.ndarray:
        """ Returns the center of the Zonotope of dimension n"""
        return self.Z[:, :1].flatten()

    @property
    def generators(self) -> np.ndarray:
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
    def order(self) -> float:
        """ Returns the order of the zonotope """
        return self.num_generators / self.dimension

    @property
    def shape(self) -> Tuple[int,int]:
        """ Returns d x n, where d is the dimensionality and n is the number of generators """
        return [self.dimension, self.num_generators]

    def copy(self) -> Zonotope:
        """ Returns a copy of the zonotope """
        return Zonotope(deepcopy(self.center)[:, np.newaxis], deepcopy(self.generators))

    def __add__(self, operand: Union[float, int, np.ndarray, Zonotope]) -> Zonotope:
        if isinstance(operand, float) or isinstance(operand, int) or isinstance(operand, np.ndarray):
            return Zonotope(self.Z[:, 0] + operand, self.Z[:, 1:])
        if isinstance(operand, Zonotope):
            assert np.all(operand.dimension == self.dimension), \
                f"Operand has not the same dimension, {self.dimension} != {operand.dimension}"
            return Zonotope(self.Z[:, 0] + operand.center, np.hstack([self.Z[:, 1:], operand.generators]))
        else:
            raise Exception(f"Addition not implemented for type {type(operand)}")

    def __mul__(self, operand: Union[int, float, np.ndarray, IntervalMatrix]) -> Zonotope:
        if isinstance(operand, float) or isinstance(operand, int):
            Z = self.Z * operand
            return Zonotope(Z[:,0], Z[:, 1:])

        elif isinstance(operand, np.ndarray):
            # Left multiplication, operand * self
            if operand.shape[1] == self.center.shape[0]:
                Z = operand @ self.Z
                return Zonotope(Z[:,0], Z[:, 1:])
            # Right multiplication, self * operand
            # It's the same as the left multiplication
            elif operand.shape[0] == self.center.shape[1]:
                return self.__mul__(operand.T)

            else:
                raise Exception('Incorrect dimension')

        elif isinstance(operand, IntervalMatrix):
            import pdb
            pdb.set_trace()

            T = operand.center
            S = operand.radius
            Zabssum = np.abs(self.Z).sum(1)
            Z = np.hstack([T @ self.Z, np.diag(S @ Zabssum)])
            return Zonotope(Z[:,0], Z[:, 1:])

        else:
            raise Exception(f"Multiplication not implemented for type {type(operand)}")


    __rmul__ = __mul__   # commutative operation
    __matmul__ = __mul__

    def __str__(self):
        return f'Center: {self.center} - Generators: {self.generators.T}'

    def sample(self, batch_size: int = 1) -> np.ndarray:
        """
        Generates a uniform random points within a zonotope

        Return a matrix of size (b x n), where n is the
        dimensionality and b is the batch size

        :param batch_size: number of random points
        :return: A matrix where each row is a point of dimension n
        """
            
        beta = np.random.uniform(low=-1, high=1, size=(self.num_generators, batch_size))
        return (self.center[:, None] + np.dot(self.generators, beta)).T

    @property
    def interval(self) -> Interval:
        """ Return the interval representation of the zonotope """
        center = self.center
        delta = np.abs(self.generators).sum(axis=1)
        return Interval(center - delta, center + delta)

    def contains(self, X: np.ndarray, tolerance: float = 1e-9) -> bool:
        """
        Return true if the zonotope contains X
        """
        assert isinstance(X, np.ndarray), 'Operand is not an array'
        return self.interval.contains(X, tolerance)

    def cartesian_product(self, W: Zonotope) -> Zonotope:
        """
        Returns the cartesian product ZxW between Z and W

        :param W: Zonotope object
        :return: Zonotope object
        """
        assert isinstance(W, Zonotope), 'Operand is not a Zonotope'
        new_center = np.hstack((self.center, W.center))
        new_generators = block_diag(self.generators, W.generators)
        return Zonotope(new_center, new_generators)

    def reduce(self, order: int) -> Zonotope:
        """
        Applies the Box reduction to the zonotope

        See also A. Girard. "Reachability of uncertain linear systems
        using zonotopes". 2005 or
        https://mediatum.ub.tum.de/doc/1379661/473275.pdf

        :param order: desired order of the zonotope
        :return: a matrix
        """
        center, G_unreduced, G_reduced = self.pick_generators(order)

        if G_reduced is not None:
            G_box = np.diag(np.abs(G_reduced).sum(axis=1))
            return Zonotope(center, np.hstack([G_unreduced, G_box]))
        
        return Zonotope(center, G_unreduced)
    
    def pick_generators(self, order: int) -> Tuple[np.ndarray, np.ndarray, Union[None,np.ndarray]]:
        """
        Select generators to be reduced.

        See also A. Girard. "Reachability of uncertain linear systems
        using zonotopes". 2005

        :param order: desired order of the zonotope
        :return center: center of reduced zonotope
        :return G_unreduced: generators that are not reduced
        :return G_reduced: generators that are reduced
        """
        center = self.center.copy()
        generators = self.generators.copy()

        generators_unreduced = generators
        generators_reduced = None

        # Reduce only if the order of the zonotope is greater than the desired order
        if self.order > order and np.any(generators):
            # Delete generators that are 0 (non zero filter)
            generators: np.ndarray = generators[:, np.any(generators, axis=0)]

            dim, num_generators = generators.shape

            h = np.linalg.norm(generators, ord=1, axis=0) - np.linalg.norm(generators, ord=np.infty, axis=0)
            n_unreduced = int(np.floor(dim * (order - 1)))
            n_reduced = num_generators - n_unreduced
            idxs = np.argpartition(h, n_reduced - 1)
            generators_reduced = generators[:, idxs[: n_reduced]]
            generators_unreduced = generators[:, idxs[n_reduced:]]
        
        return center, generators_unreduced, generators_reduced

    def over_approximate(self) -> Zonotope:
        """
        Over approximates a zonotope by a cube
        """
        delta = np.sum(np.abs(self.generators), axis=1)[:, None]
        return Zonotope(self.center, delta)

    @property
    def max_norm(self) -> Tuple[float, np.ndarray]:
        """ Returns the maximum infinity norm on the set """
        beta = cp.Variable((self.num_generators))
        y = cp.Variable(nonneg=True)
        constraints = [cp.norm(beta, p=np.infty) <= 1]

        for i in range(self.dimension):
            constraints.append(y<= self.center[i] + self.generators[i,:] @ beta)    

        problem = cp.Problem(cp.Maximize(y), constraints=constraints)

        res = problem.solve()
        return res, beta.value

    @property
    def max_num_vertices(self):
        res = 0
        for i in range(self.dimension):
            res += comb(self.num_generators - 1, i)
        return 2 * res

    def compute_vertices(self) -> np.ndarray:
        """
        Uses convex hull algorithm
        :return: an array of Vxd dimensions, where V is the number of vertices
                 and d is the dimensionality of the zonotope.
        """
        V = self.center.copy()

        for iVertex in range(self.num_generators):
            translation = self.generators[:, iVertex]
            V = np.vstack([V +  translation, V-translation])
            if iVertex > self.dimension - 1:
                try:
                    V = V[ConvexHull(V).vertices]
                except Exception as e:
                    raise('Could not compute the convex hul')
                

        return V

    def projection(self, x: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        beta = cp.Variable((self.num_generators))
        constraints = [beta <= 1, beta >= -1]

        obj = cp.norm(x - self.center - self.generators @ beta)

        problem = cp.Problem(cp.Minimize(obj), constraints=constraints)

        res = problem.solve()
        return res, self.center + self.generators @ beta.value, beta.value

        
            


            

c = np.ones((10, 1))
g = 5e-3 * np.ones((10, 1))
W = Zonotope(c, g)

Z = W * 10.0
# print(W)
# print(Z)

x = Z.sample(2)
#print(x.shape)
