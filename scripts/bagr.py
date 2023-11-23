import numpy as np
from scipy.spatial.transform import Rotation

cov = np.diag([1.1, 1.2, 1.3, 1.1, 1.2, 1.3])
R = Rotation.random().as_matrix()
big_R = np.zeros((6, 6))
big_R[:3, :3] = R
big_R[3:6, 3:6] = R
print(cov)
print(big_R)
cov1 = big_R @ cov @ big_R.T
cov2 = R @ cov[:3, :3] @ R.T

pass