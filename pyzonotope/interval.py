import numpy as np
import cvxpy as cp
from typing import Union

class Interval(object):
    """
    Interval with left and right limits
    """
    left_limit: Union[np.ndarray, cp.Expression]
    right_limit: Union[np.ndarray, cp.Expression]

    def __init__(self, left_limit: Union[float, np.ndarray, cp.Expression], right_limit: Union[float, np.ndarray, cp.Expression]):
        self.left_limit = np.array(left_limit) if not isinstance(left_limit, cp.Expression) else left_limit
        self.right_limit = np.array(right_limit) if not isinstance(right_limit, cp.Expression) else right_limit

        if self.left_limit.shape != self.right_limit.shape:
            raise ValueError('Left limit and right limit need to have the same shape')

        if isinstance(left_limit, np.ndarray) and isinstance(right_limit, np.ndarray):
            if np.any(self.left_limit >  self.right_limit):
                raise ValueError('Left limit needs cannot be greater than right limit')

    @property
    def radius(self) -> np.ndarray:
        """ Returns interval radius """
        return 0.5 * (self.right_limit - self.left_limit)

    def __str__(self):
        return f'Interval: {self.left_limit}\n{self.right_limit}'


    def contains(self, X: np.ndarray, tolerance: float = 1e-9) -> bool:
        """
        Returns true if interval contains X
        """
        assert isinstance(X, np.ndarray), "X is not a numpy array"
        assert X.shape == self.left_limit.shape, "X has not the correct shape"

        if isinstance(self.left_limit, np.ndarray) and isinstance(self.right_limit, np.ndarray):
            return np.all(X + tolerance >= self.left_limit) and np.all(X - tolerance <= self.right_limit)
        else:
            raise NotImplementedError('Contains not implemented for cvxpy expressions')


