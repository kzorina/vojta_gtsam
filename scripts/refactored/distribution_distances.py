import numpy as np
import gtsam

def mahalanobis_distance(x, covariance):
    assert (x.shape[0] == covariance.shape[0])
    assert (covariance.shape[0] == covariance.shape[1])
    cov_inv = np.linalg.inv(covariance)
    v = x
    return np.sqrt(v@cov_inv@v)

# def bhattacharyya_distance_old(x, covariance_t, covariance_x):
#     assert (x.shape[0] == covariance_t.shape[0] == covariance_x.shape[0])
#     assert (covariance_t.shape[0] == covariance_t.shape[1])
#     assert (covariance_x.shape[0] == covariance_x.shape[1])
#     cov = (covariance_t + covariance_x)/2
#     cov_inv = np.linalg.inv(cov)
#     cov_det = np.linalg.det(cov)
#     cov_t_det = np.linalg.det(covariance_t)
#     cov_x_det = np.linalg.det(covariance_x)
#     v = x
#     return (v@cov_inv@v)/8 + np.log(cov_det/(np.sqrt(cov_x_det*cov_t_det)))/2

def bhattacharyya_distance(a, b, Q_a, Q_b):
    assert (a.shape[0] == Q_a.shape[0] == Q_b.shape[0])
    assert (Q_a.shape[0] == Q_a.shape[1])
    assert (Q_b.shape[0] == Q_b.shape[1])
    Q = (Q_a + Q_b)/2
    Q_inv = np.linalg.inv(Q)
    Q_det = np.linalg.det(Q)
    Q_a_det = np.linalg.det(Q_a)
    Q_b_det = np.linalg.det(Q_b)
    v = (a - b)
    return (v@Q_inv@v)/8 + np.log(Q_det/(np.sqrt(Q_a_det*Q_b_det)))/2

def euclidean_distance(T_wa:gtsam.Pose3, T_wb:gtsam.Pose3):
    T_ab:gtsam.Pose3 = T_wa.inverse() * T_wb
    t = T_ab.translation()
    w = gtsam.Rot3.Logmap(T_ab.rotation())
    return np.linalg.norm(t) + np.linalg.norm(w)*0.625

def translation_distance(T_wa:gtsam.Pose3, T_wb:gtsam.Pose3):
    T_ab:gtsam.Pose3 = T_wa.inverse() * T_wb
    t = T_ab.translation()
    return np.linalg.norm(t)

def rotation_distance(T_wa:gtsam.Pose3, T_wb:gtsam.Pose3):
    T_ab:gtsam.Pose3 = T_wa.inverse() * T_wb
    w = gtsam.Rot3.Logmap(T_ab.rotation())
    return np.linalg.norm(w)

if __name__ == "__main__":
    T1 = gtsam.Pose3(np.eye(4))
    T2 = gtsam.Pose3(np.eye(4))
    b = euclidean_distance(T1, T2)
    print(b)