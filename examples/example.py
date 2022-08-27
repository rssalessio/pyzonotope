import numpy as np
from pyzonotope import Zonotope, MatrixZonotope, concatenate_zonotope
from pyzonotope.cvx_zonotope import CVXZonotope

dim = 5
concatenation = 10
W = Zonotope(np.random.normal(size=(dim)), np.diag(np.ones(dim)))
Z = Zonotope(np.random.normal(size=(concatenation)), np.ones((concatenation, 1)))

# Concatenate W 10 times
Mw: MatrixZonotope = concatenate_zonotope(W, concatenation)

# Multiply Mw by Z
V = Mw * Z


print(V)