__author__ = "Mederic Fourmy"

import numpy as np
import pinocchio as pin
import gtsam

# np.random.seed(2)
# pin.seed(0)

def random_cov(dim=6):
    A = np.random.rand(dim, dim)
    sig = 0.2  # meters and radians
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
T_cb_pin:pin.SE3 = pin.SE3.Random()   # estimated body pose in world frame
Q_wc_pin:np.ndarray = random_cov()
Q_cb_pin:np.ndarray = random_cov()

T_wc_gtsam:gtsam.Pose3 = gtsam.Pose3(T_wc_pin.homogeneous)
T_cb_gtsam:gtsam.Pose3 = gtsam.Pose3(T_cb_pin.homogeneous)
Q_wc_gtsam:np.ndarray = np.roll(Q_wc_pin, shift=(3, 3), axis=(0, 1))
Q_cb_gtsam:np.ndarray = np.roll(Q_cb_pin, shift=(3, 3), axis=(0, 1))

# u, s, vh = np.linalg.svd(Q_wb)

# def f(T_wc, T_cb):
#     return T_wc * T_cb

def f_pin(T_wc:pin.SE3, T_cb:pin.SE3):
    return T_wc * T_cb

def f_gtsam(T_wc:gtsam.Pose3, T_cb:gtsam.Pose3):
    return T_wc.transformPoseFrom(T_cb)

# Compute estimated body pose in camera frame
# T_wb = f(T_wc, T_cb)

T_wb_pin = f_pin(T_wc_pin, T_cb_pin)
T_wb_gtsam = f_gtsam(T_wc_gtsam, T_cb_gtsam)
# Question: what's Q_cb, the covariance of T_cb?

# Jacobians of T_cb with respect to T_wc and T_wb
# Check https://arxiv.org/abs/1812.01537 for equations (*)
# - chain rule (58)
# - inverse jacobian: (62) in general and (176) for SE(3)
# - composition jacobian (63,64) in general and (177,178) for SE(3)
# - In pinocchio, the Adjoint matrix (Ad) is called the "action"
# J_wb_wc = T_wb.action @ (-T_wc.action)  # (63) and (62)
# J_wb_cb = np.eye(6)  # (64)
# J_wb_wc = T_cb.inverse().action
# J_wb_cb = np.eye(6)
# # Chain rule
# Q_wb = J_wb_wc @ Q_wc @ J_wb_wc.T + J_wb_cb @ Q_cb @ J_wb_cb.T

J_wb_wc_pin = T_cb_pin.inverse().action
J_wb_cb_pin = np.eye(6)
Q_wb_pin = J_wb_wc_pin @ Q_wc_pin @ J_wb_wc_pin.T + J_wb_cb_pin @ Q_cb_pin @ J_wb_cb_pin.T

J_wb_wc_gtsam = T_cb_gtsam.inverse().AdjointMap()
J_wb_cb_gtsam = np.eye(6)
Q_wb_gtsam = J_wb_wc_gtsam @ Q_wc_gtsam @ J_wb_wc_gtsam.T + J_wb_cb_gtsam @ Q_cb_gtsam @ J_wb_cb_gtsam.T


# Verify numerically (Monte-Carlo)
N_samples = int(5e4)  # no difference above
nu_wb_arr = np.zeros((N_samples,6))
print(f'Monte Carlo Sampling N_samples={N_samples}')
for i in range(N_samples):
    if i % 1e4 == 0:
        print(f'{100*(i/N_samples)} %')
    T_wc_n = sample_se3(T_wc_pin, Q_wc_pin)
    T_cb_n = sample_se3(T_cb_pin, Q_cb_pin)
    T_wb_n = f_pin(T_wc_n, T_cb_n)
    nu_wb = pin.log6(T_wb_n.inverse() * T_wb_pin)  # OK
    # nu_cb = pin.log6(T_cb.inverse() * T_cb_n)  # OK 
    # nu_cb = pin.log6(T_cb_n * T_cb.inverse())  # WRONG 
    # nu_cb = pin.log6(T_cb * T_cb_n.inverse())  # WRONG 
    nu_wb_arr[i,:] = nu_wb.vector

# rowvar = each row is one observation
Q_wb_num = np.cov(nu_wb_arr, rowvar=False)


def frobenius_norm(Q1, Q2):
    # Distance measure between two matrices
    return np.sqrt(np.trace((Q1 - Q2).T @ (Q1 - Q2)))

# print('Q_wb_num')
# print(Q_wb_num)
# print('Q_cb')
# print(Q_cb_pin)
# print('Q_wb_num - Q_cb')
# print(Q_wb_num - Q_cb_pin)
print(frobenius_norm(Q_wb_num, Q_wb_pin))
print(frobenius_norm(Q_wb_pin, np.roll(Q_wb_gtsam, shift=(3, 3), axis=(0, 1))))
