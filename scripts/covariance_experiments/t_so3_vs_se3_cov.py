import numpy as np
import pinocchio as pin


# np.random.seed(2)
# pin.seed(0)
def random_cov(dim=6):
    A = np.random.rand(dim, dim)
    sig = 0.3  # meters and radians
    Q = sig**2*np.dot(A, A.T)
    return Q

def sample_se3(T: pin.SE3, Q: np.ndarray):
    assert Q.shape[0] == Q.shape[1] == 6
    # Sample tangent space element in "local" frame ("right" convention)
    eta = np.random.multivariate_normal(np.zeros(6), Q)
    return T * pin.exp6(eta)

# Create random values for all transforms:
# T = pin.SE3.Random()    # estimated camera pose in world frame
T = pin.SE3.Identity()
Q = random_cov()


# Verify numerically (Monte-Carlo)
N_samples = int(2e5)  # no difference above
nu_wb_arr = np.zeros((N_samples,6))
print(f'Monte Carlo Sampling N_samples={N_samples}')
for i in range(N_samples):
    if i % 1e4 == 0:
        print(f'{100*(i/N_samples)} %')
    T_n = sample_se3(T, Q)
    nu_wb = pin.log6(T_n)  # OK
    nu_wb_txso3 = np.zeros(6)
    nu_wb_txso3 [:3] = T_n.translation
    nu_wb_txso3 [3:6] = pin.log3(T_n.rotation)
    nu_wb_arr[i,:] = nu_wb.vector
Q_num = np.cov(nu_wb_arr, rowvar=False)

N_samples = int(2e5)  # no difference above
nu_wb_arr = np.zeros((N_samples,6))
print(f'Monte Carlo Sampling N_samples={N_samples}')
for i in range(N_samples):
    if i % 1e4 == 0:
        print(f'{100*(i/N_samples)} %')
    T_n = sample_se3(T, Q)
    nu_wb = np.zeros(6)
    nu_wb[:3] = T_n.translation
    nu_wb[3:6] = pin.log3(T_n.rotation)
    nu_wb_arr[i,:] = nu_wb
Q_num_txso3 = np.cov(nu_wb_arr, rowvar=False)


def frobenius_norm(Q1, Q2):
    # Distance measure between two matrices
    return np.sqrt(np.trace((Q1 - Q2).T @ (Q1 - Q2)))

print('Q_wb_num')
print(frobenius_norm(Q_num, Q))
print('Q_num_txso3')
print(frobenius_norm(Q_num_txso3, Q))
pass
