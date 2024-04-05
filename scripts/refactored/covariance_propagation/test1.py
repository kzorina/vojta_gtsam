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


np.random.seed(0)
pin.seed(0)

def random_cov(dim=6):
    A = np.random.rand(dim, dim)
    sig = 0.1  # meters and radians
    Q = sig**2*np.dot(A, A.T)
    return Q

def random_pose():
    T = pin.SE3.Random().homogeneous
    T[:3, 3] /= 10
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
# Q_cc = random_cov()
# Q_oo = random_cov()

Q_cc = np.eye(6)*0.001
Q_cc[:3, :3] = 0
Q_cc[5, 5] = 0.5
Q_oo = np.eye(6)*0.01
Q_oo[:3, :3] = 0


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


J_wo_wc = T_co.inverse().AdjointMap()
J_wo_co = np.eye(6)
Q_wo = J_wo_wc @ Q_cc @ J_wo_wc.T + J_wo_co @ Q_oo @ J_wo_co.T
# Q_wo = Q_cc + Q_oo

# Verify numerically (Monte-Carlo)


# fig = plt.figure()
# ax = fig.add_subplot(1,2, projection='3d')
fig, ax = plt.subplots(1,2,figsize=(10,10),subplot_kw=dict(projection='3d'))
plotter_L = Plotter(ax[0])

plotter_L.plot_T(gtsam.Pose3(np.eye(6)))
T_wc_0 = T_wc.matrix()
T_wc_0[:3, 3] = 0
J_wc_0 = gtsam.Pose3(T_wc_0).AdjointMap()
J_wc = T_wc.inverse().AdjointMap()
plotter_L.plot_Q((J_wc @ Q_cc @ J_wc.T)[3:6, 3:6], T_wc.inverse(), color='g')
# plotter.plot_Q((J_wc_0 @ Q_cc @ J_wc_0.T)[:3, :3], T_wc, color='r')
plotter_L.plot_Q((J_wc_0 @ Q_cc @ J_wc_0.T)[3:6, 3:6], T_wc, color='b')
plotter_L.plot_Q((Q_cc)[3:6, 3:6], gtsam.Pose3(np.eye(6)), color='r')
# plotter.plot_Q((Q_cc)[3:6, 3:6], T_wc, color='b')
plotter_L.plot_text(T_wc, "T_wc")
plotter_L.plot_text(T_wc.inverse(), "T_cw")
# plotter_L.plot_text(T_wc, "T_wc")
plotter_L.plot_T(T_wc.inverse())
plotter_L.plot_T(T_wc)

plotter_R = Plotter(ax[1])

plotter_R.plot_T(gtsam.Pose3(np.eye(6)))
T_wc_0 = T_wc.matrix()
T_wc_0[:3, 3] = 0
J_wc_0 = gtsam.Pose3(T_wc_0).AdjointMap()
J_wc = T_wc.inverse().AdjointMap()
plotter_R.plot_Q((J_wc_0 @ Q_cc @ J_wc_0.T)[3:6, 3:6], T_wc, color='g')
plotter_R.plot_Q((J_wc @ J_wc_0 @ Q_cc @ J_wc_0.T @ J_wc.T)[3:6, 3:6], T_wc, color='b')
plotter_R.plot_Q((Q_cc)[3:6, 3:6], gtsam.Pose3(np.eye(6)), color='r')
# plotter.plot_Q((Q_cc)[3:6, 3:6], T_wc, color='b')
plotter_R.plot_text(T_wc, "T_wc")
plotter_R.plot_T(T_wc)

# J_wo = T_wo.AdjointMap()
# plotter.plot_Q((J_wo @ Q_oo @ J_wo.T)[3:6, 3:6] * 1, T_wo, color='g')
# plotter.plot_Q((J_wo @ Q_oo @ J_wo.T)[:3, :3] * 1, T_wo, color='r')
# plotter.plot_Q((Q_oo)[:3, :3] * 1, gtsam.Pose3(np.eye(6)), color='b')
# plotter.plot_T(T_wo)
# plotter.plot_text(T_wo, "T_wo")

N_samples = int(1e3)  # no difference above
nu_wo_arr = np.zeros((N_samples,6))
points = np.zeros(((N_samples,3)))
print(f'Monte Carlo Sampling N_samples={N_samples}')
for i in range(N_samples):
    if i % 1e4 == 0:
        print(f'{100*(i/N_samples)} %')
    T_cc_n = sample_se3(Q_cc)
    T_oo_n = sample_se3(Q_oo)
    T_wo_n = T_wc * T_cc_n * T_co * T_oo_n
    # nu_wo = gtsam.Pose3.Logmap(T_wo * T_wo_n.inverse())  # local
    nu_wo = gtsam.Pose3.Logmap(T_wo.inverse() * T_wo_n)  # global
    # nu_wo = gtsam.Pose3.Logmap(T_wo_n)

    points[i,:] = (T_wc * T_cc_n).translation()

    nu_wo_arr[i,:] = nu_wo
# plotter.plot_points(points)
plt.show()
time.sleep(10)
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

