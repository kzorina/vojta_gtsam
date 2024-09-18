import gtsam
import numpy as np
from gtsam import Symbol
from gtsam.symbol_shorthand import B, V, X, L
from typing import List, Dict, Set
from SAM_noise import SAM_noise
import graphviz
import gtsam_unstable

import custom_gtsam_plot as gtsam_plot
import matplotlib.pyplot as plt

from SAM_distribution_distances import mahalanobis_distance, bhattacharyya_distance

def dX(symbol):
    return f"X({symbol - X(0)})"

def dL(symbol):
    return f"L({symbol - L(0)})"

class Landmark:
    MIN_LENGH_BEFORE_CUT = 3
    def __init__(self, symbol, frame):
        self.symbol: Symbol = symbol
        self.innitial_frame = frame
        self.number_of_detections = 1
        self.last_seen_frame = frame
        self.chain_start = symbol

    def cut_chain_tail(self, max_length = 999):
        cut = min(self.chain_start + max_length, self.symbol - Landmark.MIN_LENGH_BEFORE_CUT)
        cut = max(cut, self.chain_start)
        ret = range(self.chain_start, cut)
        self.chain_start = cut
        return ret

    def is_valid(self, current_frame):
        n = 2
        if (self.number_of_detections >= n or (current_frame - self.innitial_frame) < n) and current_frame - self.last_seen_frame < 200:
            return True
        return False

class SAM():
    def __init__(self):
        self.graph = gtsam.NonlinearFactorGraph()
        self.new_graph = gtsam.NonlinearFactorGraph()

        self.initial_estimate = gtsam.Values()
        self.current_estimate = None
        self.marginals = None

        self.new_timestamps = {}

        self.parameters = gtsam.ISAM2Params()
        self.parameters.setRelinearizeThreshold(0.1)
        self.parameters.relinearizeSkip = 1
        # self.fixed_lag_smoother: gtsam.BatchFixedLagSmoother = gtsam.BatchFixedLagSmoother(10.0)
        self.fixed_lag_smoother:gtsam_unstable.IncrementalFixedLagSmoother = gtsam_unstable.IncrementalFixedLagSmoother(30.0, self.parameters)
        # self.landmark_symbols: Dict[str:Symbol] = {}  # {object_name: symbol}
        self.detected_landmarks: Dict[str:[Symbol]] = {}
        self.landmark_count = 0
        self.all_factors_count = 0
        # self.detected_landmarks: Set = set()

        self.last_T_bc = None  # the last recorded T_bc transformation

        self.current_frame = 0

        self.camera_landmark = Landmark(X(0), 0)
        self.camera_landmark.symbol -= 1
        # self.camera_key = None

        self.K = np.array([[615.15, 0, 324.58],
                           [0, 615.25, 237.82],
                           [0, 0, 1]])
    @staticmethod
    def parse_VariableIndex(variable_index):  # TODO: REMOVE THIS TEMPORARY FIX ASAP
        string = variable_index.__repr__()
        lines = string.split('\n')
        ret = {}
        for line in lines[1:-1]:
            entries = line.split(' ')
            if entries[1][0] == 'l':
                symbol = L(int(entries[1][1:-1]))
            elif entries[1][0] == 'x':
                symbol = X(int(entries[1][1:-1]))
            else:
                raise f"invalid entry on line: {line}"
            factors = [int(i) for i in entries[2:]]
            ret[symbol] = factors
        return ret

    @staticmethod
    def count_val(vi, val):
        count = 0
        for symbol in vi:
            count += vi[symbol].count(val)
        return count

    def insert_T_bc_detection(self, T_bc: np.ndarray):
        """
        inserts camera pose estimate
        :param T_bc:
        """
        self.current_frame += 1
        # self.camera_key = X(self.current_frame)
        self.camera_landmark.symbol += 1
        pose = gtsam.Pose3(T_bc)
        noise = SAM_noise.get_panda_eef_noise()
        self.graph.add(gtsam.PriorFactorPose3(self.camera_landmark.symbol, pose, noise))
        self.new_graph.add(gtsam.PriorFactorPose3(self.camera_landmark.symbol, pose, noise))
        # if self.current_estimate != None:
        #     T_bc_1:gtsam.Pose3 = self.current_estimate.atPose3(self.camera_landmark.symbol - 1)
        #     T_cc = T_bc_1.inverse().compose(pose)
        #     self.graph.add(gtsam.BetweenFactorPose3(self.camera_landmark.symbol - 1, self.camera_landmark.symbol, T_cc, gtsam.noiseModel.Diagonal.Sigmas(1*np.ones(6))))
        #     self.new_graph.add(gtsam.BetweenFactorPose3(self.camera_landmark.symbol - 1, self.camera_landmark.symbol, T_cc, gtsam.noiseModel.Diagonal.Sigmas(1*np.ones(6))))

        self.all_factors_count += 1
        self.initial_estimate.insert(self.camera_landmark.symbol, pose)
        self.new_timestamps[self.camera_landmark.symbol] = self.current_frame
        if self.current_estimate == None:
            self.current_estimate = self.initial_estimate
        self.last_T_bc = T_bc

    def is_outlier(self, T_cn, noise:gtsam.noiseModel.Gaussian, symbol):
        T_cn: gtsam.Pose3 = gtsam.Pose3(T_cn)  # new_estimate to camera transformation
        T_bc: gtsam.Pose3 = self.current_estimate.atPose3(self.camera_landmark.symbol)  # old estimate to camera transformation
        T_bo: gtsam.Pose3 = self.current_estimate.atPose3(symbol)  # old estimate to camera transformation
        Q_oo: np.ndarray = self.marginals.marginalCovariance(symbol)  # covariance matrix of old estimate expressed in the old estimate reference frame
        T_on = T_bo.inverse().transformPoseFrom(T_bc.transformPoseFrom(T_cn))
        dist = mahalanobis_distance(gtsam.gtsam.Pose3.Logmap(T_on), np.zeros(6), Q_oo)
        print(dist)
        if dist > 10:
            return True
        return False

    def get_new_symbol(self):
        # max_idx = 10**11
        max_idx = 10**6
        symbol = L((((self.landmark_count + 1) % max_idx) * max_idx))
        return symbol

    def calculate_D(self, T_cn_s:np.ndarray, noises, object_name:str):
        """
        Calculates a 2d matrix containing distances between each new and old object estimates
        """
        T_bo_s: [gtsam.Pose3] = []  # old estimate to camera transformation
        T_bc: gtsam.Pose3 = gtsam.Pose3(self.last_T_bc)
        if object_name in self.detected_landmarks:
            for i, landmark in reversed(list(enumerate(self.detected_landmarks[object_name]))):
                T_bo_s.append(self.current_estimate.atPose3(landmark.symbol - 1))
                # T_bo_s.append(self.current_estimate.atPose3(landmark.symbol))

        D = np.ndarray((len(T_bo_s), len(T_cn_s)))
        for i in range(len(T_bo_s)):
            for j in range(len(T_cn_s)):
                Q_nn = noises[j]
                T_on = T_bo_s[i].transformPoseTo(T_bc.transformPoseFrom(gtsam.Pose3(T_cn_s[j])))
                D[i, j] = mahalanobis_distance(gtsam.Pose3.Logmap(T_on.inverse()), Q_nn.covariance())
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
            # print(minimum)
            if minimum < 20:
                assignment[i] = argmin
            padded_D[argmin, :] = np.full((padded_D.shape[1]), np.inf)
        return assignment


    def add_new_landmark(self, symbol, pose, noise, object_name):
        self.graph.add(gtsam.BetweenFactorPose3(self.camera_landmark.symbol, symbol, pose, noise))
        self.new_graph.add(gtsam.BetweenFactorPose3(self.camera_landmark.symbol, symbol, pose, noise))
        self.all_factors_count += 1

        self.detected_landmarks[object_name].append(Landmark(symbol, self.current_frame))
        # estimate = self.current_estimate.atPose3(self.camera_key).compose(pose)
        estimate = gtsam.Pose3(self.last_T_bc).compose(pose)
        self.initial_estimate.insert(symbol, estimate)
        self.new_timestamps[symbol] = float(self.current_frame)
        self.landmark_count += 1

    def insert_odometry_measurements(self):
        for object_name in self.detected_landmarks:
            for landmark in self.detected_landmarks[object_name]:
                landmark.symbol += 1
                odometry = gtsam.Pose3(np.eye(4))
                # time_elapsed = 0.000000000001
                time_elapsed = 0.000001
                time_elapsed = 0.000005
                odometry_noise = gtsam.noiseModel.Gaussian.Covariance(np.eye(6) * time_elapsed)
                self.graph.add(gtsam.BetweenFactorPose3(landmark.symbol - 1, landmark.symbol, odometry, odometry_noise))
                self.new_graph.add(gtsam.BetweenFactorPose3(landmark.symbol - 1, landmark.symbol, odometry, odometry_noise))
                self.all_factors_count += 1
                self.new_timestamps[landmark.symbol] = float(self.current_frame)
                estimate = self.current_estimate.atPose3(landmark.symbol - 1)
                self.initial_estimate.insert(landmark.symbol, estimate)

    def insert_T_co_detections(self, T_cn_s: [np.ndarray], object_name: str):
        """
        isert one or more insances of the same object type. Determines the best assignment to previous detections.
        :param T_cos: [T_co, T_co, T_co...] unordered list of objects of the same type
        """
        # T_bc = self.current_estimate.atPose3(self.camera_key)
        T_bc = gtsam.Pose3(self.last_T_bc)
        noises = []
        for j in range(len(T_cn_s)):
            noises.append(SAM_noise.get_object_in_camera_noise(T_cn_s[j], T_bc.matrix(), f=0.03455))
            # noises.append(SAM_noise.get_object_in_camera_noise_old(T_cn_s[j], T_bc.matrix(), f=0.03455))
        if object_name not in self.detected_landmarks:  # no previous instance of this object.
            self.detected_landmarks[object_name] = []
            for j in range(len(T_cn_s)):
                symbol = self.get_new_symbol()
                pose = gtsam.Pose3(T_cn_s[j])
                noise = noises[j]
                self.add_new_landmark(symbol, pose, noise, object_name)
        else:  # instance of the object has been previously detected
            D: np.ndarray = self.calculate_D(T_cn_s, noises, object_name)
            # print(object_name, D)
            assignment = self.determine_assignment(D)
            for j in range(len(T_cn_s)):
                i = assignment[j]
                pose = gtsam.Pose3(T_cn_s[j])
                noise = noises[j]
                if i == -1:  # object is too far from all other objects of the same type
                    symbol = self.get_new_symbol()
                    self.add_new_landmark(symbol, pose, noise, object_name)
                else:
                    ################ with motion model:
                    # landmark = self.detected_landmarks[object_name][i]
                    # landmark.symbol += 1
                    # landmark.number_of_detections += 1
                    # landmark.last_seen_frame = self.current_frame
                    # self.graph.add(gtsam.BetweenFactorPose3(self.camera_landmark.symbol, landmark.symbol, pose, noise))
                    # self.new_graph.add(gtsam.BetweenFactorPose3(self.camera_landmark.symbol, landmark.symbol, pose, noise))
                    # self.all_factors_count += 1
                    # odometry = gtsam.Pose3(np.eye(4))
                    # # time_elapsed = 0.000000000001
                    # sigma = 0.000001
                    # odometry_noise = gtsam.noiseModel.Gaussian.Covariance(np.eye(6) * sigma * (self.current_frame - landmark.last_seen_frame))
                    # self.graph.add(
                    #     gtsam.BetweenFactorPose3(landmark.symbol - 1, landmark.symbol, odometry, odometry_noise))
                    # self.new_graph.add(
                    #     gtsam.BetweenFactorPose3(landmark.symbol - 1, landmark.symbol, odometry, odometry_noise))
                    # self.all_factors_count += 1
                    #
                    # estimate = self.current_estimate.atPose3(landmark.symbol - 1)
                    # self.initial_estimate.insert(landmark.symbol, estimate)
                    ################ with motion model, full odometry:
                    landmark = self.detected_landmarks[object_name][i]
                    landmark.number_of_detections += 1
                    landmark.last_seen_frame = self.current_frame
                    self.graph.add(gtsam.BetweenFactorPose3(self.camera_landmark.symbol, landmark.symbol, pose, noise))
                    self.new_graph.add(gtsam.BetweenFactorPose3(self.camera_landmark.symbol, landmark.symbol, pose, noise))
                    self.all_factors_count += 1
                    ################## without motion model:
                    # landmark = self.detected_landmarks[object_name][i]
                    # self.new_graph.add(gtsam.BetweenFactorPose3(self.camera_landmark.symbol, landmark.symbol, pose, noise))
                    # self.graph.add(gtsam.BetweenFactorPose3(self.camera_landmark.symbol, landmark.symbol, pose, noise))
                    # self.new_timestamps[landmark.symbol] = self.current_frame
                    # self.all_factors_count += 1

    def update_current_estimate(self):
        self.current_estimate = self.fixed_lag_smoother.calculateEstimate()
        self.new_timestamps = {}
        self.initial_estimate.clear()
        self.new_graph = gtsam.NonlinearFactorGraph()
        for obj_name in list(self.detected_landmarks):
            for i in reversed(range(len(self.detected_landmarks[obj_name]))):
                if not self.current_estimate.exists(self.detected_landmarks[obj_name][i].symbol):
                    del self.detected_landmarks[obj_name][i]
                if len(self.detected_landmarks[obj_name]) == 0:
                    del self.detected_landmarks[obj_name]
        # self.new_graph.resize(0)

        # isam = self.fixed_lag_smoother.getISAM2()
        # for obj_name in self.detected_landmarks:
        #     for landmark in self.detected_landmarks[obj_name]:
        #         cov1 = isam.marginalCovariance(landmark.symbol)
        #         # suma = np.sum((cov1 - cov2))
        #         pass


    def update_fls(self):  # call after each change of camera pose
        self.fixed_lag_smoother.update(self.new_graph, self.initial_estimate, self.new_timestamps)

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
            for landmark in self.detected_landmarks[object_name]:
                if landmark.is_valid(self.current_frame):
                    T_bo: gtsam.Pose3 = self.current_estimate.atPose3(landmark.symbol)
                    T_bc: gtsam.Pose3 = self.current_estimate.atPose3(self.camera_landmark.symbol)
                    ret[object_name].append((T_bc.inverse().compose(T_bo)).matrix())
        return ret

    def export_current_state(self):
        ret = {}
        # marginals = gtsam.Marginals(self.graph, self.current_estimate)
        for object_name in self.detected_landmarks:
            object_entries = []
            for landmark in self.detected_landmarks[object_name]:
                entry = {}
                # cov = marginals.marginalCovariance(landmark.symbol)
                cov = self.fixed_lag_smoother.marginalCovariance(landmark.symbol)
                T:gtsam.Pose3 = self.current_estimate.atPose3(landmark.symbol)
                entry['T'] = T.matrix()
                entry['Q'] = cov
                object_entries.append(entry)
            ret[object_name] = object_entries
        return ret

    def draw_3d_estimate_mm(self):
        """Display the current estimate of a factor graph"""
        # Compute the marginals for all states in the graph.
        marginals = gtsam.Marginals(self.graph, self.current_estimate)
        # Plot the newly updated iSAM2 inference.
        fig = plt.figure(0)
        if not fig.axes:
            axes = fig.add_subplot(projection='3d')
        else:
            axes = fig.axes[0]
        plt.cla()
        for object_name in self.detected_landmarks:
            for idx, landmark in enumerate(self.detected_landmarks[object_name]):
                if landmark.is_valid(self.current_frame):
                    current_pose = self.current_estimate.atPose3(landmark.symbol)
                    name = f'{object_name[:2]}_{idx}'
                    cov = marginals.marginalCovariance(landmark.symbol)
                    gtsam_plot.plot_pose3(0, current_pose, 0.2, cov)
                    axes.text(current_pose.x(), current_pose.y(), current_pose.z(), name, fontsize=15)

        # for i in self.graph.keyVector():
        for i in range(max(self.current_frame - 10, 1), self.current_frame):
            key = X(i)
            current_pose = self.current_estimate.atPose3(key)
            name = str(Symbol(key).string())
            cov = marginals.marginalCovariance(key)
            gtsam_plot.plot_pose3(0, current_pose, 0.2, cov)
            # gtsam_plot.plot_covariance_ellipse_3d(axes, current_pose.translation(), cov[:3, :3], alpha=0.3, cmap='cool')
            axes.text(current_pose.x(), current_pose.y(), current_pose.z(), name, fontsize=15)

        ranges = (-0.8, 0.8)
        axes.set_xlim3d(ranges[0], ranges[1])
        axes.set_ylim3d(ranges[0], ranges[1])
        axes.set_zlim3d(ranges[0], ranges[1])
        fig.show()
        plt.pause(0.1)
        return fig

def main():
    pass

if __name__ == "__main__":
    main()