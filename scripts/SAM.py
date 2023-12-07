import gtsam
import numpy as np
from gtsam import Symbol
from gtsam.symbol_shorthand import B, V, X, L
from typing import List, Dict, Set
from SAM_noise import SAM_noise

# import gtsam.utils.plot as gtsam_plot
import custom_gtsam_plot as gtsam_plot
import matplotlib.pyplot as plt

from SAM_distribution_distances import mahalanobis_distance, bhattacharyya_distance
class SAM():
    def __init__(self):
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.current_estimate = None
        self.marginals = None

        self.parameters = gtsam.ISAM2Params()
        self.parameters.setRelinearizeThreshold(0.1)
        self.parameters.relinearizeSkip = 1
        self.isam = gtsam.ISAM2(self.parameters)

        self.landmark_symbols: Dict[str:Symbol] = {}  # {idx: symbol}
        self.detected_landmarks:  Dict[str:Symbol] = {}
        # self.detected_landmarks: Set = set()

        self.current_frame = 0
        self.camera_key = None

    def insert_T_bc_detection(self, T_bc: np.ndarray):
        self.current_frame += 1
        self.camera_key = X(self.current_frame)
        pose = gtsam.Pose3(T_bc)
        noise = SAM_noise.get_panda_eef_noise()
        self.graph.add(gtsam.PriorFactorPose3(self.camera_key, pose, noise))
        self.initial_estimate.insert(self.camera_key, pose)
        if self.current_estimate == None:
            self.current_estimate = self.initial_estimate
        self.update_estimate()

    def is_outlier(self, T_cn, noise:gtsam.noiseModel.Gaussian, key):
        T_cn: gtsam.Pose3 = gtsam.Pose3(T_cn)  # new_estimate to camera transformation
        T_bc: gtsam.Pose3 = self.current_estimate.atPose3(self.camera_key)  # old estimate to camera transformation
        T_bo: gtsam.Pose3 = self.current_estimate.atPose3(key)  # old estimate to camera transformation
        Q_oo: np.ndarray = self.marginals.marginalCovariance(key)  # covariance matrix of old estimate expressed in the old estimate reference frame
        Q_nn: np.ndarray = noise.covariance()  # new estimate covariance in the new estimate frame
        Q_cc: np.ndarray = self.marginals.marginalCovariance(self.camera_key)  #  camera covariance in the camera frame
        # noise_o: gtsam.noiseModel.Gaussian = gtsam.noiseModel.Gaussian.Covariance(cov_o)
        J_cb_wc_gtsam = T_wb_gtsam.inverse().AdjointMap() @ (-T_wc_gtsam.AdjointMap())
        J_cb_wb_gtsam = np.eye(6)
        # T_nc = T_cn.inverse()
        # T_on: gtsam.Pose3 = T_bo.inverse().compose(T_bc).compose(T_cn)
        # Q_nn = SAM_noise.transform_cov(T_nc, Q_cc) + Q_nn
        # Q_on = SAM_noise.transform_cov(T_on, Q_nn)
        # source: https://stats.stackexchange.com/questions/494430/covariance-of-sum-of-multivariate-normals#:~:text=If%20you%20add%20two%20multivariate,of%20the%20two%20covariance%20matrices.&text=Note%20that%20your%20v%E2%8B%85b%20is%20a%20multivariate%20normal%20distribution.
        # C_bn = C_bn +
        mahal_d = mahalanobis_distance(gtsam.gtsam.Pose3.Logmap(T_on), Q_oo, None)
        bhatt_d = bhattacharyya_distance(gtsam.gtsam.Pose3.Logmap(T_on), Q_oo, Q_on)
        if bhatt_d > 10:
            return True
        # R =
        # d = noise_o.mahalanobisDistance(T_on.translation())
        # print(d)
        # known_landmark_pose = self.current_estimate.atPose3(key)
        # known_landmark_cov = self.marginals.marginalCovariance(key)
        # new_landmark_pose = self.current_estimate.atPose3(self.camera_key).compose(pose)
        return False

    def get_landmark_key(self, idx: str):
        if idx in self.detected_landmarks:
            return self.detected_landmarks[idx]
        else:
            key = L(len(self.detected_landmarks) + 1)
            return key

    def insert_T_co_detection(self, T_co: np.ndarray, idx: str):
        key = self.get_landmark_key(idx)
        pose = gtsam.Pose3(T_co)
        T_bc = self.current_estimate.atPose3(self.camera_key).matrix()
        noise = SAM_noise.get_object_in_camera_noise(T_co, T_bc, f=0.03455)
        if idx in self.detected_landmarks:
            if not self.is_outlier(T_co, noise, key):
                self.graph.add(gtsam.BetweenFactorPose3(self.camera_key, key, pose, noise))
        else:
            self.graph.add(gtsam.BetweenFactorPose3(self.camera_key, key, pose, noise))
            self.detected_landmarks[idx] = key
            estimate = self.current_estimate.atPose3(self.camera_key).compose(pose)
            self.initial_estimate.insert(key, estimate)

    def update_estimate(self):  # call after each change of camera pose
        self.isam.update(self.graph, self.initial_estimate)
        self.current_estimate = self.isam.calculateEstimate()
        self.marginals = gtsam.Marginals(self.graph, self.current_estimate)
        self.initial_estimate.clear()

    def get_all_T_bo(self):
        ret = {}
        for idx in self.detected_landmarks:
            key = self.detected_landmarks[idx]
            pose:gtsam.Pose3 = self.current_estimate.atPose3(key)
            ret[idx] = pose.matrix()
        return ret

    def get_all_T_co(self):
        ret = {}
        for idx in self.detected_landmarks:
            key = self.detected_landmarks[idx]
            T_bo: gtsam.Pose3 = self.current_estimate.atPose3(key)
            T_bc: gtsam.Pose3 = self.current_estimate.atPose3(self.camera_key)
            ret[idx] = (T_bc.inverse().compose(T_bo)).matrix()
        return ret

    def draw_3d_estimate(self):
        """Display the current estimate of a factor graph"""
        global count
        # Compute the marginals for all states in the graph.
        marginals = gtsam.Marginals(self.graph, self.current_estimate)
        # Plot the newly updated iSAM2 inference.
        fig = plt.figure(0)
        if not fig.axes:
            axes = fig.add_subplot(projection='3d')
        else:
            axes = fig.axes[0]
        plt.cla()
        for i in self.graph.keyVector():
            current_pose = self.current_estimate.atPose3(i)
            name = str(Symbol(i).string())
            cov = marginals.marginalCovariance(i)
            gtsam_plot.plot_pose3(0, current_pose, 0.2, cov)
            gtsam_plot.plot_covariance_ellipse_3d(axes, current_pose.translation(), cov[:3, :3], alpha=0.3, cmap='cool')
            axes.text(current_pose.x(), current_pose.y(), current_pose.z(), name, fontsize=15)

        ranges = (-0.8, 0.8)
        axes.set_xlim3d(ranges[0], ranges[1])
        axes.set_ylim3d(ranges[0], ranges[1])
        axes.set_zlim3d(ranges[0], ranges[1])
        fig.show()
        keyboardClick = False
        while keyboardClick != True:
            keyboardClick = plt.waitforbuttonpress()
        # plt.pause(0.05)

def main():
    pass

if __name__ == "__main__":
    main()