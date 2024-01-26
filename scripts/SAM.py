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

        # self.landmark_symbols: Dict[str:Symbol] = {}  # {object_name: symbol}
        self.detected_landmarks: Dict[str:[Symbol]] = {}
        self.landmark_count = 0
        # self.detected_landmarks: Set = set()

        self.current_frame = 0
        self.camera_key = None

    def insert_T_bc_detection(self, T_bc: np.ndarray):
        """
        inserts camera pose estimate
        :param T_bc:
        """
        self.current_frame += 1
        self.camera_key = X(self.current_frame)
        pose = gtsam.Pose3(T_bc)
        noise = SAM_noise.get_panda_eef_noise()
        self.graph.add(gtsam.PriorFactorPose3(self.camera_key, pose, noise))
        self.initial_estimate.insert(self.camera_key, pose)
        if self.current_estimate == None:
            self.current_estimate = self.initial_estimate
        self.update_estimate()



    def is_outlier(self, T_cn, noise:gtsam.noiseModel.Gaussian, symbol):
        T_cn: gtsam.Pose3 = gtsam.Pose3(T_cn)  # new_estimate to camera transformation
        T_bc: gtsam.Pose3 = self.current_estimate.atPose3(self.camera_key)  # old estimate to camera transformation
        T_bo: gtsam.Pose3 = self.current_estimate.atPose3(symbol)  # old estimate to camera transformation
        Q_oo: np.ndarray = self.marginals.marginalCovariance(symbol)  # covariance matrix of old estimate expressed in the old estimate reference frame
        # Q_nn: np.ndarray = noise.covariance()  # new estimate covariance in the new estimate frame
        # Q_cc: np.ndarray = self.marginals.marginalCovariance(self.camera_key)  #  camera covariance in the camera frame
        # J_cb_wc_gtsam = T_wb_gtsam.inverse().AdjointMap() @ (-T_wc_gtsam.AdjointMap())
        # J_cb_wb_gtsam = np.eye(6)
        #
        # mahal_d = mahalanobis_distance(gtsam.gtsam.Pose3.Logmap(T_on), Q_oo, None)
        # bhatt_d = bhattacharyya_distance(gtsam.gtsam.Pose3.Logmap(T_on), Q_oo, Q_on)
        # if bhatt_d > 10:
        #     return True
        T_on = T_bo.inverse().transformPoseFrom(T_bc.transformPoseFrom(T_cn))
        dist = mahalanobis_distance(gtsam.gtsam.Pose3.Logmap(T_on), np.zeros(6), Q_oo)
        print(dist)
        if dist > 10:
            return True
        return False


    # def get_landmark_symbol(self, object_name: str):
    #     if object_name in self.detected_landmarks:
    #         return self.detected_landmarks[object_name]
    #     else:
    #         symbol = L(len(self.detected_landmarks) + 1)
    #         return symbol
    def get_new_symbol(self):
        symbol = L(self.landmark_count + 1)
        return symbol

    # def insert_T_co_detection(self, T_co: np.ndarray, object_name: str):
    #     symbol = self.get_landmark_symbol(object_name)
    #     pose = gtsam.Pose3(T_co)
    #     T_bc = self.current_estimate.atPose3(self.camera_key).matrix()
    #     noise = SAM_noise.get_object_in_camera_noise(T_co, T_bc, f=0.03455)
    #     if object_name in self.detected_landmarks:
    #         if not self.is_outlier(T_co, noise, symbol):
    #             self.graph.add(gtsam.BetweenFactorPose3(self.camera_key, symbol, pose, noise))
    #     else:
    #         self.graph.add(gtsam.BetweenFactorPose3(self.camera_key, symbol, pose, noise))
    #         self.detected_landmarks[object_name] = symbol
    #         estimate = self.current_estimate.atPose3(self.camera_key).compose(pose)
    #         self.initial_estimate.insert(symbol, estimate)

    def calculate_D(self, T_cn_s:np.ndarray, noises, object_name:str):
        """
        Calculates a 2d matrix containing distances between each new and old object estimates
        """
        T_bo_s: [gtsam.Pose3] = []  # old estimate to camera transformation
        Q_oo: [np.ndarray] = []  # covariance matrix of old estimate expressed in the old estimate reference frame
        T_bc: gtsam.Pose3 = self.current_estimate.atPose3(self.camera_key)
        if object_name in self.detected_landmarks:
            for symbol in self.detected_landmarks[object_name]:
                T_bo_s.append(self.current_estimate.atPose3(symbol))
                Q_oo.append(self.marginals.marginalCovariance(symbol))
        D = np.ndarray((len(T_bo_s), len(T_cn_s)))
        for i in range(len(T_bo_s)):
            for j in range(len(T_cn_s)):
                #  T_on = np.linalg.inv(T_bo_s[i]) @ T_bc @ T_cn_s[j]
                Q_cn = noises[j]
                T_on = T_bo_s[i].transformPoseTo(T_bc.transformPoseFrom(gtsam.Pose3(T_cn_s[j])))
                D[i, j] = mahalanobis_distance(gtsam.gtsam.Pose3.Logmap(T_on), Q_oo[i])
        return D

    def determine_assignment(self, D):
        assignment = [-1 for i in range(D.shape[1])]  # new_detection_idx: [old_detection_idx, ..., ...],
        # TODO: rewrite with linear programming
        # d_size = max(D.shape[0], D.shape[1])
        padded_D = np.zeros_like(D)
        # if D.shape[0] > D.shape[1]:
        #     padded_D = np.full((D.shape[0], D.shape[0]), np.inf)
        # else:
        #     padded_D = np.zeros((D.shape[1], D.shape[1]))
        padded_D[:D.shape[0], :D.shape[1]] = D
        for i in range(D.shape[1]):
            argmin = np.argmin(padded_D[:, i])
            minimum = padded_D[:, i][argmin]
            if minimum < 100:
                assignment[i] = argmin
            padded_D[argmin, :] = np.full((padded_D.shape[1]), np.inf)
        return assignment


    def add_new_landmark(self, symbol, pose, noise, object_name):
        self.graph.add(gtsam.BetweenFactorPose3(self.camera_key, symbol, pose, noise))
        self.detected_landmarks[object_name].append(symbol)
        estimate = self.current_estimate.atPose3(self.camera_key).compose(pose)
        self.initial_estimate.insert(symbol, estimate)
        self.landmark_count += 1

    def insert_T_co_detections(self, T_cn_s: [np.ndarray], object_name: str):
        """
        isert one or more insances of the same object type. Determines the best assignment to previous detections.
        :param T_cos: [T_co, T_co, T_co...] unordered list of objects of the same type
        """
        # symbol = self.get_landmark_symbol(object_name)
        # pose = gtsam.Pose3(T_co)
        # T_bc = self.current_estimate.atPose3(self.camera_key).matrix()
        # noise = SAM_noise.get_object_in_camera_noise(T_co, T_bc, f=0.03455)
        T_bc = self.current_estimate.atPose3(self.camera_key)
        noises = []
        for j in range(len(T_cn_s)):
            noises.append(SAM_noise.get_object_in_camera_noise(T_cn_s[j], T_bc.matrix(), f=0.03455))
        if object_name not in self.detected_landmarks:  # no previous instance of this object.
            self.detected_landmarks[object_name] = []
            for j in range(len(T_cn_s)):
                symbol = self.get_new_symbol()
                pose = gtsam.Pose3(T_cn_s[j])
                noise = noises[j]
                self.add_new_landmark(symbol, pose, noise, object_name)
        else:
            D: np.ndarray = self.calculate_D(T_cn_s, noises, object_name)
            assignment = self.determine_assignment(D)
            for j in range(len(T_cn_s)):
                i = assignment[j]
                pose = gtsam.Pose3(T_cn_s[j])
                noise = noises[j]
                if i == -1:
                    symbol = self.get_new_symbol()
                    self.add_new_landmark(symbol, pose, noise, object_name)
                else:
                    symbol = self.detected_landmarks[object_name][i]
                    self.graph.add(gtsam.BetweenFactorPose3(self.camera_key, symbol, pose, noise))

    def update_estimate(self):  # call after each change of camera pose
        self.isam.update(self.graph, self.initial_estimate)
        self.current_estimate = self.isam.calculateEstimate()
        self.marginals = gtsam.Marginals(self.graph, self.current_estimate)
        self.initial_estimate.clear()

    def get_all_T_bo(self):  # TODO: make compatible with duplicates
        ret = {}
        for idx in self.detected_landmarks:
            key = self.detected_landmarks[idx]
            pose:gtsam.Pose3 = self.current_estimate.atPose3(key)
            ret[idx] = pose.matrix()
        return ret

    def get_all_T_co(self):  # TODO: make compatible with duplicates
        ret = {}
        for object_name in self.detected_landmarks:
            ret[object_name] = []
            for symbol in self.detected_landmarks[object_name]:
                T_bo: gtsam.Pose3 = self.current_estimate.atPose3(symbol)
                T_bc: gtsam.Pose3 = self.current_estimate.atPose3(self.camera_key)
                ret[object_name].append((T_bc.inverse().compose(T_bo)).matrix())
        return ret

    def export_current_state(self):
        ret = {}
        marginals = gtsam.Marginals(self.graph, self.current_estimate)
        for name in self.detected_landmarks:
            object_entries = []
            for key in self.detected_landmarks[name]:
                entry = {}
                cov = marginals.marginalCovariance(key)
                T:gtsam.Pose3 = self.current_estimate.atPose3(key)
                entry['T'] = T.matrix()
                entry['Q'] = cov
                object_entries.append(entry)
            ret[name] = object_entries
        return ret


    def draw_3d_estimate(self, wait_for_interaction=False):
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
        if wait_for_interaction:
            keyboardClick = False
            while keyboardClick != True:
                keyboardClick = plt.waitforbuttonpress()
        else:
            plt.pause(0.05)

def main():
    pass

if __name__ == "__main__":
    main()