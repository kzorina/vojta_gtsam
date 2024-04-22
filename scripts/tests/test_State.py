import gtsam
import numpy as np
import unittest
# from ..refactored.State import *
from scipy.special import factorial

def extrapolate_func(dt: float, T_wo: gtsam.Pose3, derivatives: np.ndarray, Q: np.ndarray, Q_derivatives: np.ndarray):
    order = derivatives.shape[0]

    dts = np.full((order), dt)
    i = np.arange((order)) + 1
    ret = (dts ** (i*2)) / factorial(i)
    x = ret[np.newaxis].T

    tau = np.sum(derivatives * x, axis=0)
    new_Q = Q + np.sum(Q_derivatives * x[:, np.newaxis], axis=0)
    R_new = gtsam.Rot3.Expmap(tau[:3]) * T_wo.rotation()
    t_new = T_wo.translation() + tau[3:6]
    new_T_wo = gtsam.Pose3(R_new, t_new)
    return new_T_wo, new_Q

def random_cov(dim=6):
    A = np.random.rand(dim, dim)
    sig = 0.3  # meters and radians
    Q = sig**2*np.dot(A, A.T)
    Q[3:6, :3] = 0
    Q[:3, 3:6] = 0
    return Q

class TestExtrapolation(unittest.TestCase):
    def test_extrapolation(self):
        for i in range(100):
            T_wo = gtsam.Pose3.Identity()
            vel = np.random.rand((6))
            acc = np.random.rand((6))
            derivatives = np.vstack([vel, acc])
            Q = random_cov()
            Q_vel = random_cov()
            Q_acc = random_cov()
            Q_derivatives = np.stack([Q_vel, Q_acc])
            for dt in range(100):
                T_wo_new, Q_new = extrapolate_func(dt, T_wo, derivatives, Q, Q_derivatives)

                test_R_new = gtsam.Rot3.Expmap(vel[:3]*dt + (acc[:3]*dt**2)/2) * T_wo.rotation()
                test_t_new = T_wo.translation() + vel[3:6] * dt + (acc[3:6] * dt**2)/2
                test_T_wo_new = gtsam.Pose3(test_R_new, test_t_new)
                diff = gtsam.Pose3.Logmap(T_wo_new * test_T_wo_new.inverse())
                test_Q_new = Q + Q_vel*dt**2 + (Q_acc * dt**4)/2
                a, b = T_wo_new.matrix(), test_T_wo_new.matrix()
                # assert np.linalg.norm(diff) < 1e-10
                assert np.sum(test_Q_new - Q_new) < 1e-10



if __name__ == "__main__":
    unittest.main()
    print("Everything passed")