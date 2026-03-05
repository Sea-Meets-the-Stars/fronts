from matplotlib import pyplot as plt
import numpy as np


d = np.load('face4.npy')

# Set the edges to 1e-50
d[0, :] = 1e-50
d[:, 0] = 1e-50

plt.imshow(np.log10(d), origin='lower')
plt.colorbar()
plt.show()