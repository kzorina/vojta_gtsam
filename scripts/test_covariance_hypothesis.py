import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation
import custom_gtsam_plot as gtsam_plot
import matplotlib.pyplot as plt
import gtsam

# C_aa = np.array(((1, 0, 0, 0, 0, 0),
#                  (0, 1, 0, 0, 0, 0),
#                  (0, 0, 1, 0, 0, 0),
#                  (0, 0, 0, 1, 0, 0),
#                  (0, 0, 0, 0, 1, 0),
#                  (0, 0, 0, 0, 0, 1)))
# C_bb = np.array(((1, 0, 0, 0, 0, 0),
#                  (0, 2, 0, 0, 0, 0),
#                  (0, 0, 1, 0, 0, 0),
#                  (0, 0, 0, 4, 0, 0),
#                  (0, 0, 0, 0, 1, 0),
#                  (0, 0, 0, 0, 0, 5)))

def exp(a):
    T = np.zeros((4, 4))
    T[:3, :3] = pin.exp3(a[:3])
    T[:3, 3] = a[3:6]
    T[3, 3] = 1
    return T

def log(T):
    a = np.zeros(6)
    a[:3] = pin.log3(T[:3, :3])
    a[3:6] = T[:3, 3]
    return a

def transform_cov(T_ba: np.ndarray, C_ac: np.ndarray):
    assert C_ac.shape == (6, 6)
    assert T_ba.shape == (4, 4) or T_ba.shape == (3, 3)
    C_bc = C_ac
    C_bc[:3, :3] = T_ba[:3, :3] @ C_bc[:3, :3] @ T_ba[:3, :3].T
    C_bc[3:6, 3:6] = T_ba[:3, :3] @ C_bc[3:6, 3:6] @ T_ba[:3, :3].T
    return C_bc
def get_random_samples(mean, C_aa):
    return np.random.multivariate_normal(mean, C_aa, size=10000)

def get_random_rot():
    return Rotation.random().as_matrix()
def get_random_cov():
    cov = np.diag(np.random.rand(6))/100
    R = get_random_rot()
    cov = transform_cov(R, cov)
    return cov

def compare_matrices(A, B):
    return np.linalg.norm(A-B)

def test_cov_sum():
    mean = np.array((0, 0, 0, 0, 0, 0))
    fig = plt.figure(0)
    axes = fig.add_subplot(projection='3d')

    for i in range(100):

        cov_A = get_random_cov()
        cov_B = get_random_cov()
        samples_A = get_random_samples(mean, cov_A)
        samples_B = get_random_samples(mean, cov_B)

        samples_C = np.zeros_like(samples_A)
        for i in range(samples_A.shape[0]):
            T_wa = exp(samples_A[i])
            T_ab = exp(samples_B[i])
            T_wb = T_wa @ T_ab
            samples_C[i] = log(T_wb)
        cov_1 = cov_A+cov_B
        cov_2 = np.cov(samples_C.T)
        innit_axes(axes)
        draw_cov(axes, mean[3:6], cov_1[3:6, 3:6])
        draw_frame(axes, np.eye(4))
        plot_wait(plt)
        axes.clear()
        # pass
        assert compare_matrices(cov_A+cov_B, np.cov(samples_C.T)) < 2

def compose_covariances(C_aa, C_bb, T_ba):
    C_ba = transform_cov(T_ba, C_aa)
    C_b_bpa = C_bb + C_ba
    C_b_bpa[3:6, 3:6] += C_bb[:3, :3]

def skew_sym(v):
    return np.array(((0, -v[2], v[1]),
                     (v[2], 0, -v[0]),
                     (-v[1], v[0], 0)))

def test_covariance_composition():
    mean = np.array((0, 0, 0, 0, 0, 0))
    fig = plt.figure(0)
    axes = fig.add_subplot(projection='3d')

    T_ab = np.array(((1, 0, 0, 0),
                     (0, 1, 0, 0),
                     (0, 0, 1, 10),
                    (0, 0, 0, 1)))
    for i in range(100):


        # cov_A = np.array(((0.01, 0, 0, 0, 0, 0),
        #                  (0, 0.01, 0, 0, 0, 0),
        #                  (0, 0, 0.01, 0, 0, 0),
        #                  (0, 0, 0, 0, 0, 0),
        #                  (0, 0, 0, 0, 0, 0),
        #                  (0, 0, 0, 0, 0, 0)))
        # cov_B = np.array(((0, 0, 0, 0, 0, 0),
        #                  (0, 0, 0, 0, 0, 0),
        #                  (0, 0, 0, 0, 0, 0),
        #                  (0, 0, 0, 0, 0, 0),
        #                  (0, 0, 0, 0, 0, 0),
        #                  (0, 0, 0, 0, 0, 0)))
        cov_A = get_random_cov()
        cov_B = get_random_cov()
        samples_A = get_random_samples(mean, cov_A)
        samples_B = get_random_samples(mean, cov_B)

        samples_C = np.zeros_like(samples_A)
        for i in range(samples_A.shape[0]):
            T_wa = exp(samples_A[i])
            T_bb = exp(samples_B[i])
            T_wb = T_wa @ T_ab @ T_bb
            samples_C[i] = log(T_wb)
        cov_1 = cov_A+cov_B
        cov_old = cov_A+cov_B
        S = skew_sym(T_ab[:3, 3])
        rot_cov = S@cov_A[:3, :3]@S.T
        cov_1[3:6, 3:6] += rot_cov
        cov_2 = np.cov(samples_C.T)
        # np.linalg.eig((cov_2 - cov_old)[3:5, 3:5])
        # np.linalg.eig(rot_cov)
        innit_axes(axes)
        # draw_cov(axes, mean[3:6], cov_1[3:6, 3:6])
        draw_frame(axes, np.eye(4))
        plot_wait(plt)
        axes.clear()
        # pass
        assert compare_matrices(cov_A+cov_B, np.cov(samples_C.T)) < 2

def innit_axes(axes):
    ranges = (-0.8, 0.8)
    axes.set_xlim3d(ranges[0], ranges[1])
    axes.set_ylim3d(ranges[0], ranges[1])
    axes.set_zlim3d(ranges[0], ranges[1])
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')

def draw_cov(axes, mean, cov):
    gtsam_plot.plot_covariance_ellipse_3d(axes, mean, cov, alpha=0.3, cmap='hot')

def draw_frame(axes, T_wa):
    if isinstance(T_wa, np.ndarray):
        gtsam_plot.plot_pose3_on_axes(axes, gtsam.Pose3(T_wa))
    elif isinstance(T_wa, gtsam.Pose3):
        gtsam_plot.plot_pose3_on_axes(axes, T_wa)

def plot_wait(plt):
    keyboardClick = False
    while keyboardClick != True:
        keyboardClick = plt.waitforbuttonpress()

def test_cov_frame_change():
    mean = np.array((0, 0, 0, 0, 0, 0))
    for i in range(100):
        cov_A = get_random_cov()
        cov_B = get_random_cov()
        samples_A = get_random_samples(mean, cov_A)
        samples_B = get_random_samples(mean, cov_B)
        samples_C = samples_A + samples_B
        assert compare_matrices(cov_A+cov_B, np.cov(samples_C.T)) < 1

def main():
    # test_cov_sum()
    test_covariance_composition()
    # cov = np.cov(r.T)
    # print(np.cov(r))
    # print(r)
    # print(exp(r))
    pass

if __name__ == "__main__":
    main()