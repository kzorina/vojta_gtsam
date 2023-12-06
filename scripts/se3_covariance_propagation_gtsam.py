__author__ = "Mederic Fourmy"

import numpy as np
import pinocchio as pin
import gtsam


# np.random.seed(2)
# pin.seed(0)

def random_cov(dim=6):
    A = np.random.rand(dim, dim)
    sig = 0.1  # meters and radians
    Q = sig**2*np.dot(A, A.T)
    return Q

def sample_se3(T: pin.SE3, Q: np.ndarray):
    assert Q.shape[0] == Q.shape[1] == 6
    # Sample tangent space element in "local" frame ("right" convention)
    eta = np.random.multivariate_normal(np.zeros(6), Q)
    return T * pin.exp6(eta)

def sample_se3_gtsam(T: gtsam.Pose3, Q: np.ndarray):
    assert Q.shape[0] == Q.shape[1] == 6
    eta = np.random.multivariate_normal(np.zeros(6), Q)
    return T.transformPoseFrom(gtsam.Pose3.Expmap(np.roll(eta, 3)))

# Create random values for all transforms:
T_wc_pin = pin.SE3.Random()    # estimated camera pose in world frame
T_wb_pin = pin.SE3.Random()    # estimated body pose in world frame
Q_wc = random_cov()
Q_wb = random_cov()

u, s, vh = np.linalg.svd(Q_wb)

def f(T_wc, T_wb):
    return T_wc.inverse() * T_wb

# Compute estimated body pose in camera frame
T_cb = f(T_wc, T_wb)
# Question: what's Q_cb, the covariance of T_cb?

# Jacobians of T_cb with respect to T_wc and T_wb
# Check https://arxiv.org/abs/1812.01537 for equations (*)
# - chain rule (58)
# - inverse jacobian: (62) in general and (176) for SE(3)
# - composition jacobian (63,64) in general and (177,178) for SE(3)
# - In pinocchio, the Adjoint matrix (Ad) is called the "action"
J_cb_wc = T_wb.inverse().action @ (-T_wc.action)  # (63) and (62)
J_cb_wb = np.eye(6)  # (64)

# Chain rule
Q_cb = J_cb_wc @ Q_wc @ J_cb_wc.T + J_cb_wb @ Q_wb @ J_cb_wb.T


# Verify numerically (Monte-Carlo)
N_samples = int(1e5)  # no difference above
nu_cb_arr = np.zeros((N_samples,6))
print(f'Monte Carlo Sampling N_samples={N_samples}')
for i in range(N_samples):
    if i % 1e4 == 0:
        print(f'{100*(i/N_samples)} %')
    T_wc_n = sample_se3(T_wc, Q_wc)
    T_wb_n = sample_se3(T_wb, Q_wb)
    T_cb_n = f(T_wc_n, T_wb_n)
    nu_cb = pin.log6(T_cb_n.inverse() * T_cb)  # OK
    # nu_cb = pin.log6(T_cb.inverse() * T_cb_n)  # OK
    # nu_cb = pin.log6(T_cb_n * T_cb.inverse())  # WRONG
    # nu_cb = pin.log6(T_cb * T_cb_n.inverse())  # WRONG
    nu_cb_arr[i,:] = nu_cb.vector

# rowvar = each row is one observation
Q_cb_num = np.cov(nu_cb_arr, rowvar=False)


def frobenius_norm(Q1, Q2):
    # Distance measure between two matrices
    return np.sqrt(np.trace((Q1 - Q2).T @ (Q1 - Q2)))

print('Q_cb_num')
print(Q_cb_num)
print('Q_cb')
print(Q_cb)
print('Q_cb_num - Q_cb')
print(Q_cb_num - Q_cb)
print(frobenius_norm(Q_cb_num, Q_wb))
print(frobenius_norm(Q_cb_num, Q_wc))
print(frobenius_norm(Q_cb_num, Q_cb))
