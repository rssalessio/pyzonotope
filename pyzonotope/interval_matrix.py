import numpy as np
from pyzonotope.interval import Interval

class IntervalMatrix(object):
    """
    Represents a matrix interval
    """
    dim: int
    inf: np.ndarray
    sup: np.ndarray
    _interval: np.ndarray
    matrix_center = np.ndarray
    setting: str = 'sharpivmult'

    def __init__(self, matrix_center: np.ndarray, matrix_delta: np.ndarray, setting: str = None):
        self.dim = len(matrix_center)
        
        _matrix_delta = np.abs(matrix_delta)
        self.inf = matrix_center - _matrix_delta
        self.sup = matrix_center + _matrix_delta

        self._interval = Interval(self.inf, self.sup)
        self.matrix_center = matrix_center
        
        if isinstance(setting, str):
            self.setting = setting

    def __str__(self):
        return f'Interval matrix: sup {self.sup}\ninf {self.inf}'

    @property
    def center(self) -> np.ndarray:
        return self.matrix_center

    @property
    def radius(self) -> np.ndarray:
        return self.interval.radius

    @property
    def interval(self) -> Interval:
        """ Returns the interval representation """
        return self._interval

    def contains(self, X: np.ndarray, tolerance: float = 1e-9) -> bool:
        """
        Returns true if the interval matrix contains X
        """
        return self.interval.contains(X, tolerance)
