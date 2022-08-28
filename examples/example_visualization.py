from pyzonotope import Zonotope
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt

dim = 2
W = Zonotope([0, 0], [[1, 0.5, -0.3], [-0.5, 0.7, 0.1]])

fig, ax = plt.subplots()

collection = PatchCollection([W.polygon], alpha=0.4, color='red')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

ax.add_collection(collection)
plt.grid()
plt.show()