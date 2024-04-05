import gtsam
import numpy as np
import unittest
from ..refactored.State import *

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
                T_wo_new, Q_new = BareTrack.extrapolate_func(dt, T_wo, derivatives, Q, Q_derivatives)

                test_R_new = gtsam.Rot3.Expmap(vel[:3]*dt + (acc[:3]*dt**2)/2) * T_wo.rotation()
                test_t_new = T_wo.translation() + vel[3:6] * dt + (acc[3:6] * dt**2)/2
                test_T_wo_new = gtsam.Pose3(test_R_new, test_t_new)
                diff = gtsam.Pose3.Logmap(T_wo_new * test_T_wo_new.inverse())
                a, b = T_wo_new.matrix(), test_T_wo_new.matrix()
                assert np.linalg.norm(diff) < 1e-10



if __name__ == "__main__":
    unittest.main()
    print("Everything passed")