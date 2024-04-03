import time

import numpy as np
import pinocchio as pin
import gtsam
import sys
import os
from pathlib import Path
from ScenePlotter import Plotter
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, RadioButtons


# np.random.seed(2)
# pin.seed(0)

def random_cov(dim=6):
    A = np.random.rand(dim, dim)
    sig = 0.2  # meters and radians
    Q = sig**2*np.dot(A, A.T)
    return Q

def random_pose():
    return gtsam.Pose3(pin.SE3.Random().homogeneous)

# def sample_se3(T: pin.SE3, Q: np.ndarray):
#     assert Q.shape[0] == Q.shape[1] == 6
#     # Sample tangent space element in "local" frame ("right" convention)
#     eta = np.random.multivariate_normal(np.zeros(6), Q)
#     return T * pin.exp6(eta)

def sample_se3(Q: np.ndarray):
    assert Q.shape[0] == Q.shape[1] == 6
    eta:np.ndarray = np.random.multivariate_normal(np.zeros(6), Q)
    return gtsam.Pose3.Expmap(eta)

# Create random values for all transforms:


T_wc:gtsam.Pose3 = random_pose()
T_co:gtsam.Pose3 = random_pose()
Q_cc = random_cov()
Q_oo = random_cov()


# Compute estimated body pose in camera frame
# T_wb = f(T_wc, T_cb)

T_wo = T_wc * T_co
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

# J_wb_wc_pin = T_cb_pin.inverse().action
# J_wb_cb_pin = np.eye(6)
# Q_wb_pin = J_wb_wc_pin @ Q_wc_pin @ J_wb_wc_pin.T + J_wb_cb_pin @ Q_cb_pin @ J_wb_cb_pin.T

# J_wo_wc = T_co.inverse().AdjointMap()
# J_wo_co = np.eye(6)
# Q_wo = J_wo_wc @ Q_wc @ J_wo_wc.T + J_wo_co @ Q_co @ J_wo_co.T


# Verify numerically (Monte-Carlo)
N_samples = int(2e4)  # no difference above
nu_wo_arr = np.zeros((N_samples,6))
print(f'Monte Carlo Sampling N_samples={N_samples}')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plotter = Plotter(ax)
for i in range(N_samples):
    if i % 1e4 == 0:
        print(f'{100*(i/N_samples)} %')
    T_cc_n = sample_se3(Q_cc)
    T_oo_n = sample_se3(Q_oo)
    T_wo_n = T_wc * T_cc_n * T_co * T_oo_n
    nu_wo = gtsam.Pose3.Logmap(T_wo.inverse() * T_wo_n)

    J_wc = T_wc.inverse().AdjointMap()
    plotter.plot_Q((J_wc @ Q_cc @ J_wc.T)[3:6, 3:6]*0.1, T_wc)
    plotter.plot_T(T_wc)
    J_wo = T_wo.inverse().AdjointMap()
    plotter.plot_Q((J_wo @ Q_oo @ J_wo.T)[3:6, 3:6]*0.1, T_wc)
    plotter.plot_T(T_wc)
    plt.show()
    time.sleep(10)
    # plotter.set_camera_view()

    nu_wo_arr[i,:] = nu_wo


# rowvar = each row is one observation
Q_wo_num = np.cov(nu_wo_arr, rowvar=False)

def frobenius_norm(Q1, Q2):
    # Distance measure between two matrices
    return np.sqrt(np.trace((Q1 - Q2).T @ (Q1 - Q2)))

# print('Q_wb_num')
# print(Q_wb_num)
# print('Q_cb')
# print(Q_cb_pin)
# print('Q_wb_num - Q_cb')
# print(Q_wb_num - Q_cb_pin)
print(frobenius_norm(Q_wo_num, Q_wo))

