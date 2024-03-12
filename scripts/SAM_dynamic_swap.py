import gtsam
import numpy
import numpy as np
from gtsam import Symbol
from gtsam.symbol_shorthand import B, V, X, L
from typing import List, Dict, Set
from SAM_noise import SAM_noise
import graphviz
import gtsam_unstable
from functools import partial
from custom_odom_factors import error_velocity_integration_local, error_velocity_integration_global

import custom_gtsam_plot as gtsam_plot
import matplotlib.pyplot as plt
from collections import defaultdict

from SAM_distribution_distances import mahalanobis_distance, bhattacharyya_distance

def dX(symbol):
    return symbol - X(0)

def dL(symbol):
    return symbol - L(0)

def dV(symbol):
    return symbol - V(0)

class Landmark:
    MIN_LENGH_BEFORE_CUT = 3

    def __init__(self, symbol, frame):
        self.symbol: Symbol = symbol
        self.innitial_frame = frame
        self.number_of_detections = 1
        self.last_seen_frame = frame
        self.chain_start = symbol
        self.initial_symbol = symbol

    def cut_chain_tail(self, max_length = 999):
        cut = min(self.chain_start + max_length, self.symbol - Landmark.MIN_LENGH_BEFORE_CUT)
        cut = max(cut, self.chain_start)
        ret = range(self.chain_start, cut)
        self.chain_start = cut
        return ret

    def is_valid(self, current_frame, Q, t_validity_treshold = 0.00001, R_validity_treshold = 0.0002):
        R_det = np.linalg.det(Q[:3, :3]) ** 0.5
        t_det = np.linalg.det(Q[3:6, 3:6]) ** 0.5

        if t_det > t_validity_treshold or R_det > R_validity_treshold:
            return False
        return True

        # if t_det > 0.0000004 or R_det > 0.000008:
        # n = 2
        # if (self.number_of_detections >= n or (current_frame - self.innitial_frame) < n) and current_frame - self.last_seen_frame < 200:
        #     return True
        # return False

class ISAM2_wrapper:
    def __init__(self):
        self.marginals = None
        self.chunk_count = 2
        self.active_chunk = 0
        self.parameters = gtsam.ISAM2Params()
        self.parameters.setRelinearizeThreshold(0.1)
        self.parameters.relinearizeSkip = 1
        self.isams = []
        self.initial_estimate = gtsam.Values()
        self.current_estimate = None
        self.new_graph = gtsam.NonlinearFactorGraph()
        self.active_graphs = []
        for i in range(self.chunk_count):
            self.isams.append(gtsam.ISAM2(self.parameters))
            self.active_graphs.append(gtsam.NonlinearFactorGraph())

    def add_factor(self, factor):
        self.new_graph.add(factor)
        for i in range(self.chunk_count):
            self.active_graphs[i].add(factor)

    def inser_estimate(self, symbol, pose):
        self.initial_estimate.insert(symbol, pose)

    def swap_chunks(self, landmarks):
        new_graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()
        for label in landmarks:
            for idx in range(len(landmarks[label])):
                landmark = landmarks[label][idx]
                pose = self.current_estimate.atPose3(landmark.symbol)
                Q_pose = self.isams[self.active_chunk].marginalCovariance(landmark.symbol)
                pose_noise = gtsam.noiseModel.Gaussian.Covariance(Q_pose)
                new_graph.add(gtsam.PriorFactorPose3(landmark.symbol, pose, pose_noise))
                initial_estimate.insert(landmark.symbol, pose)
                velocity_symbol = V(dL(landmark.symbol-1))
                if self.current_estimate.exists(velocity_symbol):
                    nu = self.current_estimate.atVector(velocity_symbol)
                    Q_velocity = self.isams[self.active_chunk].marginalCovariance(velocity_symbol)
                else:
                    nu = np.zeros(6)
                    Q_velocity = np.eye(6)*10
                velocity_noise = gtsam.noiseModel.Gaussian.Covariance(Q_velocity)
                new_graph.add(gtsam.PriorFactorVector(velocity_symbol, nu, velocity_noise))
                initial_estimate.insert(velocity_symbol, nu)
        self.active_graphs[self.active_chunk] = self.new_graph.clone()
        self.isams[self.active_chunk] = gtsam.ISAM2(self.parameters)
        self.isams[self.active_chunk].update(new_graph, initial_estimate)
        self.active_chunk = (self.chunk_count + 1) % self.chunk_count

    def update(self):
        for isam in self.isams:
            isam.update(self.new_graph, self.initial_estimate)
        self.current_estimate = self.isams[self.active_chunk].calculateEstimate()
        self.initial_estimate.clear()
        self.new_graph = gtsam.NonlinearFactorGraph()
        # self.marginals = gtsam.Marginals(self.active_graphs[self.active_chunk], self.current_estimate)

    def marginalCovariance(self, symbol):
        return self.isams[self.active_chunk].marginalCovariance(symbol)
        # return self.marginals.marginalCovariance(symbol)


class SAM:
    def __init__(self):

        self.isam_wrapper = ISAM2_wrapper()

        self.detected_landmarks: Dict[str:[Symbol]] = {}
        self.landmark_count = 0
        self.all_factors_count = 0
        # self.detected_landmarks: Set = set()

        self.last_T_bc = None  # the last recorded T_bc transformation

        self.current_frame = 0
        self.current_time_stamp = 0
        self.previous_time_stamp = 0

        self.camera_landmark = Landmark(X(0), 0)
        self.camera_landmark.symbol -= 1
        # self.camera_key = None

        self.K = np.array([[615.15, 0, 324.58],
                           [0, 615.25, 237.82],
                           [0, 0, 1]])

        self.SYMBOL_GAP = 10**6

        self.outlier_rejection_treshold = 20
        self.t_validity_treshold = 0.00001
        self.R_validity_treshold = 0.0002
        self.cov1 = 0.001
        self.cov2 = 0.0001
        self.window_size = 20
        self.chunk_size = 10

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

    def insert_T_bc_detection(self, T_bc: np.ndarray, timestamp):
        """
        inserts camera pose estimate
        :param T_bc:
        """
        self.current_frame += 1
        self.previous_time_stamp = self.current_time_stamp
        self.current_time_stamp = timestamp
        # self.camera_key = X(self.current_frame)
        self.camera_landmark.symbol += 1
        pose = gtsam.Pose3(T_bc)
        noise = SAM_noise.get_panda_eef_noise()
        self.isam_wrapper.add_factor(gtsam.PriorFactorPose3(self.camera_landmark.symbol, pose, noise))
        self.isam_wrapper.inser_estimate(self.camera_landmark.symbol, pose)
        # self.current_graph.add(gtsam.PriorFactorPose3(self.camera_landmark.symbol, pose, noise))
        # self.initial_estimate.insert(self.camera_landmark.symbol, pose)

        self.all_factors_count += 1
        # self.new_timestamps[self.camera_landmark.symbol] = self.current_frame
        # if self.current_estimate == None:
        #     self.current_estimate = self.initial_estimate
        self.last_T_bc = T_bc

    def get_new_symbol(self):
        # max_idx = 10**11
        max_idx = self.SYMBOL_GAP
        symbol = L((((self.landmark_count + 1) % max_idx) * max_idx))
        return symbol

    def landmarkAtPose3(self, landmark):
        T_bo_0 = self.isam_wrapper.current_estimate.atPose3(landmark.symbol - 1)
        twist_symbol = V(dL(landmark.symbol - 1))
        if self.isam_wrapper.current_estimate.exists(twist_symbol):
            nu = self.isam_wrapper.current_estimate.atVector(V(dL(landmark.symbol - 1)))
        else:
            nu = numpy.zeros(6)
        T_bb = gtsam.Pose3.Expmap(nu * self.get_dt())
        T_bo = T_bb * T_bo_0
        return T_bo

    def get_dt(self):
        return (self.current_time_stamp - self.previous_time_stamp)


    def calculate_D(self, T_cn_s:np.ndarray, noises, object_name:str):
        """
        Calculates a 2d matrix containing distances between each new and old object estimates
        """
        T_bo_s: [gtsam.Pose3] = []  # old estimate to camera transformation
        T_bc: gtsam.Pose3 = gtsam.Pose3(self.last_T_bc)
        if object_name in self.detected_landmarks:
            for i, landmark in enumerate(self.detected_landmarks[object_name]):
                T_bo = self.landmarkAtPose3(landmark)
                T_bo_s.append(T_bo)

        D = np.ndarray((len(T_bo_s), len(T_cn_s)))
        for i in range(len(T_bo_s)):
            for j in range(len(T_cn_s)):
                Q_nn = noises[j]
                T_on = T_bo_s[i].transformPoseTo(T_bc.transformPoseFrom(gtsam.Pose3(T_cn_s[j])))
                w = gtsam.Pose3.Logmap(T_on.inverse())
                D[i, j] = mahalanobis_distance(w, Q_nn.covariance())
        return D

    def determine_assignment(self, D):
        assignment = [-1 for i in range(D.shape[1])]  # new_detection_idx: [old_detection_idx, ..., ...],
        padded_D = np.zeros_like(D)
        padded_D[:D.shape[0], :D.shape[1]] = D
        for i in range(D.shape[1]):
            argmin = np.argmin(padded_D[:, i])
            minimum = padded_D[:, i][argmin]
            if minimum < self.outlier_rejection_treshold:
                assignment[i] = argmin
            padded_D[argmin, :] = np.full((padded_D.shape[1]), np.inf)
        return assignment


    def add_new_landmark(self, symbol, pose, noise, object_name):
        self.isam_wrapper.add_factor(gtsam.BetweenFactorPose3(self.camera_landmark.symbol, symbol, pose, noise))
        T_bo_estimate = gtsam.Pose3(self.last_T_bc).compose(pose)
        self.isam_wrapper.inser_estimate(symbol, T_bo_estimate)
        self.all_factors_count += 1
        self.detected_landmarks[object_name].append(Landmark(symbol, self.current_frame))
        self.landmark_count += 1

    def insert_odometry_measurements(self):
        for object_name in self.detected_landmarks:
            for landmark in self.detected_landmarks[object_name]:

                if landmark.initial_symbol < landmark.symbol:  # landmark is not new
                    prior_cst_twist = gtsam.noiseModel.Isotropic.Sigma(6, self.cov1 * self.get_dt())
                    self.isam_wrapper.add_factor(gtsam.BetweenFactorVector(V(dL(landmark.symbol-1)), V(dL(landmark.symbol)), np.zeros(6), prior_cst_twist))

                landmark.symbol += 1

                prior_int_twist = gtsam.noiseModel.Isotropic.Sigma(6, self.cov2 * self.get_dt())
                error_func = partial(error_velocity_integration_global, self.get_dt())
                twist_symbol = V(dL(landmark.symbol - 1))
                fint = gtsam.CustomFactor(
                    prior_int_twist,
                    [landmark.symbol - 1, landmark.symbol, twist_symbol],
                    error_func
                )
                self.isam_wrapper.add_factor(fint)
                T_oo_estimate = self.isam_wrapper.current_estimate.atPose3(landmark.symbol - 1)
                self.isam_wrapper.inser_estimate(landmark.symbol, T_oo_estimate)
                self.isam_wrapper.inser_estimate(twist_symbol, np.zeros(6))
                # ### less dirty hack
                if landmark.initial_symbol == landmark.symbol - 1:
                    bogus_noise = gtsam.noiseModel.Isotropic.Sigma(6, 10.0)
                    self.isam_wrapper.add_factor(gtsam.PriorFactorVector(twist_symbol, np.zeros(6), bogus_noise))
                self.all_factors_count += 1


    def insert_T_co_detections(self, T_cn_s: [np.ndarray], object_name: str, px_counts = None):
        """
        isert one or more insances of the same object type. Determines the best assignment to previous detections.
        :param T_cos: [T_co, T_co, T_co...] unordered list of objects of the same type
        """
        noises = []
        for j in range(len(T_cn_s)):
            px_count = px_counts[j]
            noises.append(SAM_noise.get_object_in_camera_noise_px(T_cn_s[j], px_count))
        if object_name not in self.detected_landmarks:  # no previous instance of this object.
            self.detected_landmarks[object_name] = []
            for j in range(len(T_cn_s)):
                symbol = self.get_new_symbol()
                pose = gtsam.Pose3(T_cn_s[j])
                noise = noises[j]
                self.add_new_landmark(symbol, pose, noise, object_name)
        else:  # instance of the object has been previously detected
            D: np.ndarray = self.calculate_D(T_cn_s, noises, object_name)
            assignment = self.determine_assignment(D)
            for j in range(len(T_cn_s)):
                i = assignment[j]
                pose = gtsam.Pose3(T_cn_s[j])
                noise = noises[j]
                if i == -1:  # object is too far from all other objects of the same type
                    symbol = self.get_new_symbol()
                    self.add_new_landmark(symbol, pose, noise, object_name)
                else:
                    landmark = self.detected_landmarks[object_name][i]
                    landmark.number_of_detections += 1
                    landmark.last_seen_frame = self.current_frame
                    self.isam_wrapper.add_factor(gtsam.BetweenFactorPose3(self.camera_landmark.symbol, landmark.symbol, pose, noise))
                    self.all_factors_count += 1

    def update_fls(self):  # call after each change of camera pose
        self.isam_wrapper.update()
        if self.current_frame % self.chunk_size == self.chunk_size-1:
            self.isam_wrapper.swap_chunks(self.detected_landmarks)
        for label in list(self.detected_landmarks):
            for idx in reversed(range(len(self.detected_landmarks[label]))):
                landmark = self.detected_landmarks[label][idx]
                if (self.current_frame - landmark.last_seen_frame) > self.window_size:
                    del self.detected_landmarks[label][idx]
            if len(self.detected_landmarks[label]) == 0:
                del self.detected_landmarks[label]
        # self.current_graph.resize(self.current_graph.size() - len(entries[0]))

    def get_all_T_bo(self):  # TODO: make compatible with duplicates
        ret = {}
        for idx in self.detected_landmarks:
            key = self.detected_landmarks[idx]
            pose:gtsam.Pose3 = self.current_estimate.atPose3(key)
            ret[idx] = pose.matrix()
        return ret

    def get_all_T_co(self, timestamp=None, current_T_bc = None):  # TODO: make compatible with duplicates
        ret = {}
        for object_name in self.detected_landmarks:
            ret[object_name] = []
            for landmark in self.detected_landmarks[object_name]:
                Q = self.isam_wrapper.marginalCovariance(landmark.symbol)
                landmark_valid = landmark.is_valid(self.current_frame, Q, self.t_validity_treshold, self.R_validity_treshold)
                # if landmark.is_valid(self.current_frame, Q):
                if current_T_bc is None:
                    T_bc: gtsam.Pose3 = self.isam_wrapper.current_estimate.atPose3(self.camera_landmark.symbol)
                else:
                    T_bc: gtsam.Pose3 = gtsam.Pose3(current_T_bc)
                T_bo: gtsam.Pose3 = self.isam_wrapper.current_estimate.atPose3(landmark.symbol)
                if timestamp is not None and landmark.initial_symbol != landmark.symbol:
                    dt = timestamp - self.current_time_stamp
                    nu12 = self.isam_wrapper.current_estimate.atVector(V(dL(landmark.symbol - 1)))
                    T_bb = gtsam.Pose3.Expmap(nu12*dt)
                else:
                    T_bb = gtsam.Pose3.Identity()
                T_co = (T_bc.inverse().compose(T_bb).compose(T_bo)).matrix()
                ret[object_name].append({"T_co":T_co, "id":landmark.initial_symbol,"Q":Q, "valid":landmark_valid}, )
        return ret

    def export_current_state(self):
        ret = {}
        # marginals = gtsam.Marginals(self.graph, self.current_estimate)
        for object_name in self.detected_landmarks:
            object_entries = []
            for landmark in self.detected_landmarks[object_name]:
                entry = {}
                # cov = marginals.marginalCovariance(landmark.symbol)
                cov = self.isam_wrapper.current_estimate.marginalCovariance(landmark.symbol)
                T:gtsam.Pose3 = self.isam_wrapper.current_estimate.atPose3(landmark.symbol)
                entry['T'] = T.matrix()
                entry['Q'] = cov
                object_entries.append(entry)
            ret[object_name] = object_entries
        return ret

    # def draw_3d_estimate_mm(self):
    #     """Display the current estimate of a factor graph"""
    #     # Compute the marginals for all states in the graph.
    #     marginals = gtsam.Marginals(self.current_graph, self.current_estimate)
    #     # Plot the newly updated iSAM2 inference.
    #     fig = plt.figure(0)
    #     if not fig.axes:
    #         axes = fig.add_subplot(projection='3d')
    #     else:
    #         axes = fig.axes[0]
    #     plt.cla()
    #     for object_name in self.detected_landmarks:
    #         for idx, landmark in enumerate(self.detected_landmarks[object_name]):
    #             if landmark.is_valid(self.current_frame, self.marginals.marginalCovariance(landmark.symbol), self.t_validity_treshold, self.R_validity_treshold):
    #                 current_pose = self.current_estimate.atPose3(landmark.symbol)
    #                 name = f'{object_name[:2]}_{idx}'
    #                 cov = 500*marginals.marginalCovariance(landmark.symbol)
    #                 gtsam_plot.plot_pose3(0, current_pose, 0.2, cov)
    #                 axes.text(current_pose.x(), current_pose.y(), current_pose.z(), name, fontsize=15)
    #
    #     # for i in self.graph.keyVector():
    #     for i in range(max(self.current_frame - 10, 1), self.current_frame):
    #         key = X(i)
    #         current_pose = self.current_estimate.atPose3(key)
    #         name = str(Symbol(key).string())
    #         cov = marginals.marginalCovariance(key)
    #         gtsam_plot.plot_pose3(0, current_pose, 0.2, cov)
    #         # gtsam_plot.plot_covariance_ellipse_3d(axes, current_pose.translation(), cov[:3, :3], alpha=0.3, cmap='cool')
    #         axes.text(current_pose.x(), current_pose.y(), current_pose.z(), name, fontsize=15)
    #
    #     ranges = (-0.8, 0.8)
    #     axes.set_xlim3d(ranges[0], ranges[1])
    #     axes.set_ylim3d(ranges[0], ranges[1])
    #     axes.set_zlim3d(ranges[0], ranges[1])
    #     fig.show()
    #     plt.pause(0.1)
    #     return fig

def main():
    pass

if __name__ == "__main__":
    main()