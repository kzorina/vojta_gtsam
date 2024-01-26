import numpy as np
import pinocchio as pin
import gtsam
from scipy.spatial.transform import Rotation


def sample_se3():
    rot = Rotation.random().as_matrix()
    trans = np.random.rand(3)
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = trans
    return T

def test_T(T_np:np.ndarray, T_pin:pin.SE3, T_gtsam:gtsam.Pose3):
    T_np_np = T_np
    T_np_pin = T_pin.homogeneous
    T_np_gtsam = T_gtsam.matrix()
    assert np.allclose(T_np_np, T_np_pin)
    assert np.allclose(T_np_np, T_np_gtsam)

def compare_pin_gtsam(T_pin, T_gtsam):
    T_np_pin = T_pin.homogeneous
    T_np_gtsam = T_gtsam.matrix()
    assert np.allclose(T_np_pin, T_np_gtsam)

def test_composition():
    T_np_wa:np.ndarray = sample_se3()
    T_pin_wa:pin.SE3 = pin.SE3(T_np_wa)
    T_gtsam_wa:gtsam.Pose3 = gtsam.Pose3(T_np_wa)

    T_np_ab:np.ndarray = sample_se3()
    T_pin_ab:pin.SE3 = pin.SE3(T_np_ab)
    T_gtsam_ab:gtsam.Pose3 = gtsam.Pose3(T_np_ab)

    T_np_wb:np.ndarray = T_np_wa @ T_np_ab
    T_pin_wb:pin.SE3 = (T_pin_wa * T_pin_ab)
    T_gtsam_wb:gtsam.Pose3 = T_gtsam_wa.transformPoseFrom(T_gtsam_ab)
    test_T(T_np_wb, T_pin_wb, T_gtsam_wb)

    T_np_ab2:np.ndarray = np.linalg.inv(T_np_wa) @ T_np_wb
    T_pin_ab2:pin.SE3 = (T_pin_wa.inverse() * T_pin_wb)
    T_gtsam_ab2:gtsam.Pose3 = T_gtsam_wa.transformPoseTo(T_gtsam_wb)
    T_gtsam2_ab2:gtsam.Pose3 = T_gtsam_wa.inverse().compose(T_gtsam_wb)
    test_T(T_np_ab2, T_pin_ab2, T_gtsam_ab2)

def random_cov(dim=6):
    A = np.random.rand(dim, dim)
    sig = 0.1  # meters and radians
    Q = sig**2*np.dot(A, A.T)
    return Q

def test_exp6():
    Q = random_cov()
    # Q_gtsam = Q[::-1,::-1].T
    eta = np.random.multivariate_normal(np.zeros(6), Q)
    # eta_gtsam = np.random.multivariate_normal(np.zeros(6), Q_gtsam)
    T_pin = pin.exp6(eta)
    T_gtsam = gtsam.Pose3.Expmap(np.roll(eta, 3))
    compare_pin_gtsam(T_pin, T_gtsam)
    #  pin.exp6(eta) ~ gtsam.Pose3.Expmap(np.roll(eta, 3))
    #  pin.exp6(np.roll(eta, 3)) ~ gtsam.Pose3.Expmap(eta)

def test_rolled_covariance():
    N = 10000
    Q = random_cov()
    # Q_gtsam = Q[::-1, ::-1].T
    Q_gtsam = np.roll(Q, shift=(3, 3), axis=(0, 1))
    etas = np.random.multivariate_normal(np.zeros(6), Q, size=N)
    etas_gtsam = np.random.multivariate_normal(np.zeros(6), Q_gtsam, size=N)
    etas_gtsam2 = np.roll(etas_gtsam, shift=3, axis=1)
    np_Q = np.cov(etas.T)
    np_Q_gtsam = np.cov(etas_gtsam2.T)
    pass

def main():
    test_rolled_covariance()
    test_composition()
    test_exp6()

if __name__ == "__main__":
    main()