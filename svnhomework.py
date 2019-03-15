import numpy as np
import matplotlib.pyplot as plt

#plus = np.array([[2, 2], [3, 3], [3, 0]])
#minus = np.array([[0, 0], [2, 0], [0, 2]])

plus = np.array([[0,1],[2,3]])
minus = np.array([[0, 3], [2, 1]])

plt.scatter(plus[:, 0], plus[:, 1], marker='+', s=200)
plt.scatter(minus[:, 0], minus[:, 1], marker='D', s=150)

plt.show()
