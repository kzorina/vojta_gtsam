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
    sig = 0.4  # meters and radians
    Q = sig**2*np.dot(A, A.T)
    return Q

def random_pose():
    T = pin.SE3.Random().homogeneous
    T[:3, 3] /= 10
    return gtsam.Pose3(pin.SE3.Random().homogeneous)

def sample_se3(Q: np.ndarray):
    assert Q.shape[0] == Q.shape[1] == 6
    eta:np.ndarray = np.random.multivariate_normal(np.zeros(6), Q)
    return gtsam.Pose3.Expmap(eta)

# Create random values for all transforms:


T_wc:gtsam.Pose3 = random_pose()
T_co:gtsam.Pose3 = random_pose()
T_co = T_co.matrix()
T_co[:3, 3] += np.array((0, 0, 1)) * 10000.0
T_co = gtsam.Pose3(T_co)
Q_cc = random_cov()
Q_oo = random_cov()
Q_oo = np.zeros((6, 6))
a = Q_cc[:3, :3]
Q_cc = np.zeros((6, 6))
Q_cc[:3, :3] = a

# Q_cc = np.eye(6)*0.001
# Q_cc[:3, :3] = 0
# Q_cc[5, 5] = 0.5
# Q_oo = np.eye(6)*0.01
# Q_oo[:3, :3] = 0

def skew_sym(v):
    return np.array(((0, -v[2], v[1]),
                     (v[2], 0, -v[0]),
                     (-v[1], v[0], 0)))

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

# https://gtsam.org/2021/02/23/uncertainties-part2.html
# https://arxiv.org/pdf/1906.07795.pdf

# J_wc_0 = gtsam.Pose3(T_wc_0).AdjointMap()
# J_wc = T_wc.inverse().AdjointMap()
# plotter_R.plot_Q((J_wc_0 @ Q_cc @ J_wc_0.T)[3:6, 3:6], T_wc, color='g')


Q_ww_1 = T_wc.AdjointMap() @ (Q_cc + T_co.AdjointMap() @ Q_oo @ T_co.AdjointMap().T) @ T_wc.AdjointMap().T
S = skew_sym(T_co.translation())
Q_co = np.zeros((6, 6))
Q_co[3:6, 3:6] = S @ Q_cc[:3, :3] @ S.T
Q_ww_2 = T_wc.AdjointMap() @ (Q_cc + T_co.AdjointMap() @ Q_oo @ T_co.AdjointMap().T + Q_co) @ T_wc.AdjointMap().T
Q_ww_3 = T_wc.AdjointMap() @ Q_cc @ T_wc.AdjointMap().T + T_wo.AdjointMap() @ Q_oo @ T_wo.AdjointMap().T

fig, ax = plt.subplots(1,2,figsize=(10,10),subplot_kw=dict(projection='3d'))

N_samples = int(1e3)  # no difference above
nu_ww_arr = np.zeros((N_samples,6))
# points = np.zeros(((N_samples,3)))
print(f'Monte Carlo Sampling N_samples={N_samples}')
for i in range(N_samples):
    if i % 1e4 == 0:
        print(f'{100*(i/N_samples)} %')
    T_cc_n = sample_se3(Q_cc)
    T_oo_n = sample_se3(Q_oo)
    T_ww_n = T_wc * T_cc_n * T_co * T_oo_n * T_wo.inverse()
    a = T_ww_n.matrix()
    b = (T_wc * T_cc_n * T_co * T_oo_n).matrix()
    c = T_wo.matrix()
    # nu_wo = gtsam.Pose3.Logmap(T_wo * T_wo_n.inverse())  # local
    nu_ww = gtsam.Pose3.Logmap(T_ww_n)  # global
    # nu_wo = gtsam.Pose3.Logmap(T_wo_n)

    # points[i,:] = (T_wc * T_cc_n).translation()

    nu_ww_arr[i,:] = nu_ww
Q_ww_num = np.cov(nu_ww_arr, rowvar=False)

def frobenius_norm(Q1, Q2):
    # Distance measure between two matrices
    return np.sqrt(np.trace((Q1 - Q2).T @ (Q1 - Q2)))

print(f"1: {frobenius_norm(Q_ww_num, Q_ww_1)}")
print(f"2: {frobenius_norm(Q_ww_num, Q_ww_2)}")
print('')
