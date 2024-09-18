import gtsam
import numpy
import numpy as np
from gtsam import Symbol
from gtsam.symbol_shorthand import B, V, X, L
from typing import List, Dict, Set
from SAM_noise import SAM_noise
# import graphviz
import gtsam_unstable
from functools import partial
import custom_odom_factors

import custom_gtsam_plot as gtsam_plot
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass

from SAM_distribution_distances import mahalanobis_distance, bhattacharyya_distance

def dX(symbol):
    return symbol - X(0)

def dL(symbol):
    return symbol - L(0)

def dV(symbol):
    return symbol - V(0)

class Landmark:

    def __init__(self, symbol, frame, settings=None):
        self.symbol: Symbol = symbol
        self.innitial_frame = frame
        self.number_of_detections = 1
        self.last_seen_frame = frame
        self.chain_start = symbol
        self.initial_symbol = symbol
        self.hysteresis_active = False
        self.settings = settings

    def is_valid(self, current_frame, Q):
        R_det = np.linalg.det(Q[:3, :3]) ** 0.5
        t_det = np.linalg.det(Q[3:6, 3:6]) ** 0.5

        if self.hysteresis_active:
            if t_det < self.settings.t_validity_treshold * self.settings.hysteresis_coef and R_det < self.settings.R_validity_treshold * self.settings.hysteresis_coef:
                return True
            else:
                self.hysteresis_active = False
                return False
        if t_det < self.settings.t_validity_treshold and R_det < self.settings.R_validity_treshold:
            self.hysteresis_active = True
            return True
        return False

        # if t_det > 0.0000004 or R_det > 0.000008:
        # n = 2
        # if (self.number_of_detections >= n or (current_frame - self.innitial_frame) < n) and current_frame - self.last_seen_frame < 200:
        #     return True
        # return False

@dataclass
class SAMSettings:
    mod: int = 1
    cov_drift_lin_vel: float = 0.1
    cov_drift_ang_vel: float = 0.1
    cov2_t: float = 0.0001
    cov2_R: float = 0.0001
    t_validity_treshold: float = 0.00001
    R_validity_treshold: float = 0.0002
    window_size: int = 20
    chunk_size: int = 10
    outlier_rejection_treshold: float = 40
    velocity_prior_sigma: float = 10
    velocity_diminishing_coef:float = 0.99
    hysteresis_coef:float = 1
    reject_overlaps:float = 0.0

    def __repr__(self):
        return f"{self.window_size}_" \
               f"{self.hysteresis_coef}_" \
               f"{self.cov_drift_lin_vel}_" \
               f"{self.cov_drift_ang_vel}_" \
               f"{self.cov2_t:.2E}_" \
               f"{self.cov2_R:.2E}_" \
               f"{self.outlier_rejection_treshold}_" \
               f"{self.t_validity_treshold:.2E}_" \
               f"{self.R_validity_treshold:.2E}"

class SymbolQueue:
    def __init__(self):
        self.MAX_AGE = 20
        self.timestamps = defaultdict(list)
        self.first_timestamp = 0
        self.current_timestamp = 0
        self.last_factor = -1
        self.factors = defaultdict(list)


    def push_factor(self, symbols):
        self.last_factor += 1
        for symbol in symbols:
            if symbol not in self.factors:
                self.timestamps[self.current_timestamp].append(symbol)
            self.factors[symbol].append(self.last_factor)

    def pop(self):
        """
        :return: all symbols and all factors that are older than self.MAX_AGE
        """
        if self.MAX_AGE > self.current_timestamp - self.first_timestamp:
            return [[],[]]
        ret = [[], []]
        for symbol in self.timestamps[self.first_timestamp]:
            if symbol in self.factors:
                ret[1].append(symbol)
            ret[0] += self.factors[symbol]
            del self.factors[symbol]
        del self.timestamps[self.first_timestamp]
        self.first_timestamp += 1
        return ret
    def increment(self):
        self.current_timestamp += 1

class ISAM2_wrapper:
    def __init__(self, settings:SAMSettings):
        self.marginals = None

        self.parameters = gtsam.LevenbergMarquardtParams()
        self.initial_estimate = gtsam.Values()
        self.current_estimate = None
        self.new_graph = gtsam.NonlinearFactorGraph()
        self.symbol_queue = SymbolQueue()
        self.symbol_queue.MAX_AGE = settings.window_size
        self.current_graph = gtsam.NonlinearFactorGraph()
        self.active_graphs = []
        self.SYMBOL_GAP = 10**6

    def add_factor(self, factor):
        self.current_graph.add(factor)
        self.symbol_queue.push_factor(factor.keys())

    def inser_estimate(self, symbol, pose):
        self.initial_estimate.insert(symbol, pose)


    def update(self, detected_landmarks, current_frame):
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.current_graph, self.initial_estimate, self.parameters)
        self.current_estimate = optimizer.optimize()
        self.marginals = gtsam.Marginals(self.current_graph, self.current_estimate)
        #  remove oldest factors and variables:
        entries = self.symbol_queue.pop()
        for factor in entries[0]:
            self.current_graph.remove(factor)
        for symbol in entries[1]:
            self.initial_estimate.erase(symbol)
            if 0 < dV(symbol) < 10 ** 4 * self.SYMBOL_GAP:  # symbol belongs to a velocity variable
                twist_symbol = symbol + 1
                bogus_noise = gtsam.noiseModel.Isotropic.Sigma(6, 100.0)
                twist_estimate = self.current_estimate.atVector(twist_symbol)
                self.current_graph.add(gtsam.PriorFactorVector(twist_symbol, twist_estimate, bogus_noise))  # adding this prior factor was deemed necesarry to prevent unconstrained system.
                self.symbol_queue.push_factor([twist_symbol])

        # forget all objects that have not been seen for a while and delete entire chains.
        self.symbol_queue.increment()
        for obj_name in list(detected_landmarks):  # forget all objects that have not been seen for a while.
            for i in reversed(range(len(detected_landmarks[obj_name]))):
                landmark = detected_landmarks[obj_name][i]
                symbol = landmark.symbol
                if (current_frame - landmark.last_seen_frame) >= self.symbol_queue.MAX_AGE:  # object is too old, completly remove it
                    twist_symbol = V(dL(symbol - 1))
                    while twist_symbol in self.symbol_queue.factors:
                        for factor in self.symbol_queue.factors[twist_symbol]:
                            self.current_graph.remove(factor)
                        del self.symbol_queue.factors[twist_symbol]
                        self.initial_estimate.erase(twist_symbol)
                        twist_symbol -= 1
                    while symbol in self.symbol_queue.factors:
                        for factor in self.symbol_queue.factors[symbol]:
                            self.current_graph.remove(factor)
                        del self.symbol_queue.factors[symbol]
                        self.initial_estimate.erase(symbol)
                        symbol -= 1
                    del detected_landmarks[obj_name][i]
                if len(detected_landmarks[obj_name]) == 0:
                    del detected_landmarks[obj_name]

    def marginalCovariance(self, symbol):
        # return self.isams[self.active_chunk].marginalCovariance(symbol)
        return self.marginals.marginalCovariance(symbol)


class SAM:
    def __init__(self, settings):

        self.isam_wrapper = ISAM2_wrapper(settings)

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

        self.settings = settings


    @staticmethod
    def parse_VariableIndex(variable_index):
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

    # def extrapolate_T_bo(self, T_bo_0, nu, dt):
    #     a = self.settings.velocity_diminishing_coef
    #     if isinstance(T_bo_0, np.ndarray):
    #         T_bb = gtsam.Pose3.Expmap(nu*(a ** dt - 1)/ np.log(a)).matrix()
    #         return T_bb @ T_bo_0
    #     # else:
    #     elif isinstance(T_bo_0, gtsam.Pose3):
    #         T_bb = gtsam.Pose3.Expmap(nu*(a ** dt - 1)/ np.log(a))
    #         return T_bb * T_bo_0
    #     else:
    #         raise Exception(f"T_bo_0 has invalid type{type(T_bo_0)}, must be either np.ndarray or gtsam.Pose3")
    def extrapolate_T_bo(self, T_bo_0, nu, dt):
        if isinstance(T_bo_0, np.ndarray):
            return custom_odom_factors.plus_so3r3_global(gtsam.Pose3(T_bo_0), nu, dt).matrix()
        elif isinstance(T_bo_0, gtsam.Pose3):
            return custom_odom_factors.plus_so3r3_global(T_bo_0, nu, dt)
        else:
            raise Exception(f"T_bo_0 has invalid type{type(T_bo_0)}, must be either np.ndarray or gtsam.Pose3")

    def landmarkAtPose3(self, landmark):
        T_bo_0:gtsam.Pose3 = self.isam_wrapper.current_estimate.atPose3(landmark.symbol - 1)
        twist_symbol = V(dL(landmark.symbol - 1))
        if self.isam_wrapper.current_estimate.exists(twist_symbol):
            nu = self.isam_wrapper.current_estimate.atVector(V(dL(landmark.symbol - 1)))
        else:
            nu = numpy.zeros(6)
        T_bo = self.extrapolate_T_bo(T_bo_0=T_bo_0, nu=nu, dt=self.get_dt())
        # T_bb = gtsam.Pose3.Expmap(nu * self.get_dt())
        # T_bo = T_bb * T_bo_0
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
            if minimum < self.settings.outlier_rejection_treshold:
                assignment[i] = argmin
            padded_D[argmin, :] = np.full((padded_D.shape[1]), np.inf)
        return assignment


    def add_new_landmark(self, symbol, pose, noise, object_name):
        self.isam_wrapper.add_factor(gtsam.BetweenFactorPose3(self.camera_landmark.symbol, symbol, pose, noise))
        T_bo_estimate = gtsam.Pose3(self.last_T_bc).compose(pose)
        self.isam_wrapper.inser_estimate(symbol, T_bo_estimate)
        self.all_factors_count += 1
        self.detected_landmarks[object_name].append(Landmark(symbol, self.current_frame, self.settings))
        self.landmark_count += 1

    def insert_odometry_measurements(self):
        for object_name in self.detected_landmarks:
            for landmark in self.detected_landmarks[object_name]:

                if landmark.initial_symbol < landmark.symbol:  # landmark is not new
                    cov1 = np.eye(6)
                    cov1[:3, :3] = np.eye(3) * self.settings.cov_drift_ang_vel * self.get_dt()
                    cov1[3:6, 3:6] = np.eye(3) * self.settings.cov_drift_lin_vel * self.get_dt()

                    prior_cst_twist = gtsam.noiseModel.Gaussian.Covariance(cov1)
                    # prior_cst_twist = gtsam.noiseModel.Isotropic.Sigma(6, self.settings.cov1 * self.get_dt())
                    self.isam_wrapper.add_factor(gtsam.BetweenFactorVector(V(dL(landmark.symbol-1)), V(dL(landmark.symbol)), np.zeros(6), prior_cst_twist))

                landmark.symbol += 1

                cov2 = np.eye(6)
                cov2[:3, :3] = np.eye(3) * self.settings.cov2_R * self.get_dt()
                cov2[3:6, 3:6] = np.eye(3) * self.settings.cov2_t * self.get_dt()
                prior_int_twist = gtsam.noiseModel.Gaussian.Covariance(cov2)
                error_func = partial(custom_odom_factors.error_velocity_integration_so3r3_global, self.get_dt())
                # error_func = partial(custom_odom_factors.error_velocity_integration_global, self.get_dt())
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
                    bogus_noise = gtsam.noiseModel.Isotropic.Sigma(6, self.settings.velocity_prior_sigma)
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
        self.isam_wrapper.update(self.detected_landmarks, self.current_frame)
        # for label in list(self.detected_landmarks):
        #     for idx in reversed(range(len(self.detected_landmarks[label]))):
        #         landmark = self.detected_landmarks[label][idx]
        #         if (self.current_frame - landmark.last_seen_frame) > self.settings.window_size:
        #             del self.detected_landmarks[label][idx]
        #     if len(self.detected_landmarks[label]) == 0:
        #         del self.detected_landmarks[label]
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
                landmark_valid = landmark.is_valid(self.current_frame, Q)
                # if landmark.is_valid(self.current_frame, Q):
                if current_T_bc is None:
                    T_bc: gtsam.Pose3 = self.isam_wrapper.current_estimate.atPose3(self.camera_landmark.symbol)
                else:
                    T_bc: gtsam.Pose3 = gtsam.Pose3(current_T_bc)
                T_bo: gtsam.Pose3 = self.isam_wrapper.current_estimate.atPose3(landmark.symbol)
                T_wo = T_bo.matrix()
                if timestamp is not None and landmark.initial_symbol != landmark.symbol:
                    dt = timestamp - self.current_time_stamp
                    nu12 = self.isam_wrapper.current_estimate.atVector(V(dL(landmark.symbol - 1)))
                    T_bb = gtsam.Pose3.Expmap(nu12*dt)
                else:
                    T_bb = gtsam.Pose3.Identity()
                T_co = (T_bc.inverse().compose(T_bb).compose(T_bo)).matrix()

                if landmark_valid and self.settings.reject_overlaps > 0:
                    for obj_inst in range(len(ret[object_name])):
                        if ret[object_name][obj_inst]["valid"]:
                            other_T_wo = ret[object_name][obj_inst]["T_wo"]
                            other_Q = ret[object_name][obj_inst]["Q"]
                            dist = np.linalg.norm(T_wo[:3, 3] - other_T_wo[:3, 3])
                            if dist < self.settings.reject_overlaps:
                                if np.linalg.det(Q) > np.linalg.det(other_Q):
                                    landmark_valid = False
                                else:
                                    ret[object_name][obj_inst]["valid"] = False

                ret[object_name].append({"T_co":T_co, "T_wo":T_wo, "id":landmark.initial_symbol,"Q":Q, "valid":landmark_valid}, )
        return ret

    def export_current_state(self):
        ret = {}
        # marginals = gtsam.Marginals(self.graph, self.current_estimate)
        for object_name in self.detected_landmarks:
            object_entries = []
            for landmark in self.detected_landmarks[object_name]:
                Q = self.isam_wrapper.marginalCovariance(landmark.symbol)
                landmark_valid = landmark.is_valid(self.current_frame, Q)
                entry = {}
                T:gtsam.Pose3 = self.isam_wrapper.current_estimate.atPose3(landmark.symbol)
                entry['T_bo'] = T.matrix()
                entry['Q'] = Q
                entry['valid'] = landmark_valid
                entry['id'] = landmark.initial_symbol
                if (landmark.symbol > landmark.initial_symbol):
                    entry['nu'] = self.isam_wrapper.current_estimate.atVector(V(dL(landmark.symbol - 1)))
                else:
                    entry['nu'] = np.zeros(6)
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