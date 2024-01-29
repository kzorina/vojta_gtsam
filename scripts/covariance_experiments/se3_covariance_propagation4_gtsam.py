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

# def sample_se3_gtsam(T: gtsam.Pose3, Q: np.ndarray):
#     assert Q.shape[0] == Q.shape[1] == 6
#     eta = np.random.multivariate_normal(np.zeros(6), Q)
#     return T.transformPoseFrom(gtsam.Pose3.Expmap(np.roll(eta, 3)))

# Create random values for all transforms:
T_wa_pin:pin.SE3 = pin.SE3.Identity()    # estimated camera pose in world frame
T_ba_pin:pin.SE3 = pin.SE3.Random()   # estimated body pose in world frame
T_ba_pin.translation = np.array([0, 0, 0])
# T_ba_pin = pin.SE3(np.array(((1, 0, 0, 0),
#                              (0, 1, 0, 0),
#                              (0, 0, 1, 0),
#                             (0, 0, 0, 1))))
# T_wo_pin:pin.SE3 = pin.SE3.Random()   # estimated body pose in world frame

# Q_aa_pin:np.ndarray = random_cov()
Q_aa_pin:np.ndarray = np.diag(np.array([1, 1, 1, 0.1, 0.1, 0.1]))

def f_pin(T_wa:pin.SE3, T_aa:pin.SE3):
    return T_wa * T_aa
def transform_cov(T_ba:np.ndarray, C_ac:np.ndarray):
    assert C_ac.shape == (6, 6)
    if isinstance(T_ba, gtsam.Pose3):
        T_ba = T_ba.matrix()
    assert T_ba.shape == (4, 4) or T_ba.shape == (3, 3)
    # J = np.zeros((6, 6))
    x = pin.SE3.Identity()
    x.rotation = T_ba
    J = x.action
    J[:3, :3] = T_ba[:3, :3]
    J[3:6, 3:6] = T_ba[:3, :3]
    C_bc = np.zeros_like(C_ac)
    C_bc[:3, :3] = T_ba[:3, :3] @ C_ac[:3, :3] @ T_ba[:3, :3].T
    C_bc[3:6, 3:6] = T_ba[:3, :3] @ C_ac[3:6, 3:6] @ T_ba[:3, :3].T
    C_bc2 = J @ C_ac @ J.T
    return C_bc2

# J = np.zeros((6, 6))
# J[:3, :3] = T_ab_pin.rotation
# J[3:6, 3:6] = T_ab_pin.rotation
Q_ba_pin = transform_cov(T_ba_pin.rotation, Q_aa_pin)

# Verify numerically (Monte-Carlo)
N_samples = int(1e5)  # no difference above
nu_ba_arr = np.zeros((N_samples,6))
print(f'Monte Carlo Sampling N_samples={N_samples}')
for i in range(N_samples):
    if i % 1e4 == 0:
        print(f'{100*(i/N_samples)} %')
    T_aa_n = sample_se3(T_wa_pin, Q_aa_pin)
    T_ba_n = T_ba_pin * T_aa_n
    # T_on_n = f_pin(T_wc_n, T_cn_n, T_wo_pin)
    # nu_on = pin.log6(T_on_n*T_on_pin.inverse())
    nu_ba = pin.log6(T_ba_n)
    nu_ba_arr[i,:] = nu_ba.vector

# rowvar = each row is one observation
Q_ba_num = np.cov(nu_ba_arr, rowvar=False)

def frobenius_norm(Q1, Q2):
    # Distance measure between two matrices
    return np.sqrt(np.trace((Q1 - Q2).T @ (Q1 - Q2)))

# print('Q_wb_num')
# print(Q_wb_num)
# print('Q_cb')
# print(Q_cb_pin)
# print('Q_wb_num - Q_cb')
# print(Q_wb_num - Q_cb_pin)
print(frobenius_norm(Q_ba_num, Q_ba_pin))
pass
# print(frobenius_norm(Q_wb_pin, np.roll(Q_wb_gtsam, shift=(3, 3), axis=(0, 1))))
