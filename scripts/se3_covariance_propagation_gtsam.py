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
T_wc_pin:pin.SE3 = pin.SE3.Random()    # estimated camera pose in world frame
T_wb_pin:pin.SE3 = pin.SE3.Random()    # estimated body pose in world frame
Q_wc_pin:np.ndarray = random_cov()
Q_wb_pin:np.ndarray = random_cov()

T_wc_gtsam:gtsam.Pose3 = gtsam.Pose3(T_wc_pin.homogeneous)
T_wb_gtsam:gtsam.Pose3 = gtsam.Pose3(T_wb_pin.homogeneous)
Q_wc_gtsam:np.ndarray = np.roll(Q_wc_pin, shift=(3, 3), axis=(0, 1))
Q_wb_gtsam:np.ndarray = np.roll(Q_wb_pin, shift=(3, 3), axis=(0, 1))

# u, s, vh = np.linalg.svd(Q_wb)

def f_pin(T_wc:pin.SE3, T_wb:pin.SE3):
    return T_wc.inverse() * T_wb

def f_gtsam(T_wc:gtsam.Pose3, T_wb:gtsam.Pose3):
    return T_wc.inverse().transformPoseFrom(T_wb)
    # return T_wc.inverse() * T_wb

# Compute estimated body pose in camera frame
T_cb_pin = f_pin(T_wc_pin, T_wb_pin)
T_cb_gtsam = f_gtsam(T_wc_gtsam, T_wb_gtsam)
# Question: what's Q_cb, the covariance of T_cb?

# Jacobians of T_cb with respect to T_wc and T_wb
# Check https://arxiv.org/abs/1812.01537 for equations (*)
# - chain rule (58)
# - inverse jacobian: (62) in general and (176) for SE(3)
# - composition jacobian (63,64) in general and (177,178) for SE(3)
# - In pinocchio, the Adjoint matrix (Ad) is called the "action"
J_cb_wc_pin = T_wb_pin.inverse().action @ (-T_wc_pin.action)  # (63) and (62)
J_cb_wb_pin = np.eye(6)  # (64)

# J_cb_wc_gtsam = T_wb_gtsam.inverse().action @ (-T_wc_gtsam.action)  # (63) and (62)
J_cb_wc_gtsam = T_wb_gtsam.inverse().AdjointMap() @ (-T_wc_gtsam.AdjointMap())
J_cb_wb_gtsam = np.eye(6)  # (64)

# Chain rule
Q_cb_pin = J_cb_wc_pin @ Q_wc_pin  @ J_cb_wc_pin.T + J_cb_wb_pin  @ Q_wb_pin  @ J_cb_wb_pin.T
Q_cb_gtsam = J_cb_wc_gtsam @ Q_wc_gtsam  @ J_cb_wc_gtsam.T + J_cb_wb_gtsam  @ Q_wb_gtsam  @ J_cb_wb_gtsam.T


# Verify numerically (Monte-Carlo)
N_samples = int(1e4)  # no difference above
nu_cb_arr = np.zeros((N_samples,6))
print(f'Monte Carlo Sampling N_samples={N_samples}')
for i in range(N_samples):
    if i % 1e4 == 0:
        print(f'{100*(i/N_samples)} %')
    T_wc_n = sample_se3(T_wc_pin, Q_wc_pin)
    T_wb_n = sample_se3(T_wb_pin, Q_wb_pin)
    T_cb_n = f_pin(T_wc_n, T_wb_n)
    nu_cb = pin.log6(T_cb_n.inverse() * T_cb_pin)  # OK
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
print(Q_cb_pin)
print('Q_cb_num - Q_cb')
print(Q_cb_num - Q_cb_pin)
print(frobenius_norm(Q_cb_num, Q_wb_pin))
print(frobenius_norm(Q_cb_num, Q_wc_pin))
print(frobenius_norm(Q_cb_num, Q_cb_pin))
