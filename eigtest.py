import numpy as np

a=np.array([[2,3],[3,-1]])
eigval,eigvector=np.linalg.eig(a)

print(a)
print(eigval)
print(eigvector)