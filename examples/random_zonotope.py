from pyzonotope import Zonotope
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np

dim = np.random.randint(low=4, high=8)
num_generators = np.random.randint(low=5, high=10)
W = Zonotope(np.zeros(dim), np.random.normal(size=(dim, num_generators)))

print(f'Dimension: {dim} - Number of generators {num_generators} - Order: {W.order} - Number of vertices: {len(W.compute_vertices())}')
