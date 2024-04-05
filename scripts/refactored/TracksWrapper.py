from typing import List, Dict, Set
import gtsam
import numpy as np
from gtsam import Symbol
from gtsam.symbol_shorthand import B, V, X, L
from FactorGraphWrapper import FactorGraphWrapper
from GlobalParams import GlobalParams
from distribution_distances import mahalanobis_distance, bhattacharyya_distance
from collections import defaultdict
import custom_odom_factors
from functools import partial
import utils

class Camera:
    def __init__(self, parent):
        self.parent: Tracks = parent
        self.num_detected = 0
        self.last_seen_time_stamp = None
        self.last_update_stamp = -1

        self.cached_attributes = {"T_wo_init": [-1, None], # init means without extrapolation
                                  "Q_init":[-1, None]}

    def get_symbol(self):
        symbol = X(self.parent.SYMBOL_GAP + self.num_detected)
        return symbol

    def add_detection(self, T_wc:gtsam.Pose3, Q, time_stamp):
        T_wc = gtsam.Pose3(T_wc)
        self.num_detected += 1
        self.last_seen_time_stamp = time_stamp
        noise = gtsam.noiseModel.Gaussian.Covariance(Q)
        factor = gtsam.PriorFactorPose3(self.get_symbol(), gtsam.Pose3(T_wc), noise)
        self.parent.factor_graph.add_factor(factor)
        self.parent.factor_graph.inser_estimate(self.get_symbol(), T_wc)
        self.cached_attributes["T_wo_init"][0] = self.parent.update_stamp
        self.cached_attributes["T_wo_init"][1] = T_wc

    def get_T_wc_init(self) -> gtsam.Pose3:
        attribute_name = "T_wo_init"
        if self.cached_attributes[attribute_name][0] < self.parent.update_stamp:
            self.cached_attributes[attribute_name][0] = self.parent.update_stamp
            self.cached_attributes[attribute_name][1] = None
        return self.cached_attributes[attribute_name][1]

    def __repr__(self):
        return f"idx:Camera, num_detected:{self.num_detected}, last_seen_time_stamp:{self.last_seen_time_stamp}"

class Track:
    def __init__(self, parent, idx, obj_label):
        self.obj_label = obj_label
        self.parent:Tracks = parent
        self.idx = idx
        self.num_detected = 0  # how many times has this track been detected and added to the factor graph
        self.last_seen_time_stamp = None
        self.last_update_stamp = -1

        self.cached_attributes = {"T_wo_init": [-1, None],  # init means without extrapolation
                                  "Q_init": [-1, None],
                                  "nu_init": [-1, None]}
        self.max_order_derivative = 1
    def is_valid(self, time_stamp):
        Q = self.get_Q_extrapolated(time_stamp)
        R_det = np.linalg.det(Q[:3, :3]) ** 0.5
        t_det = np.linalg.det(Q[3:6, 3:6]) ** 0.5
        a, b = np.linalg.det(Q[:3, :3]), np.linalg.det(Q[3:6, 3:6])
        if t_det < self.parent.params.t_validity_treshold and R_det < self.parent.params.R_validity_treshold:
            return True
        return False

    def get_symbol(self):
        symbol = L(self.parent.SYMBOL_GAP * self.idx + self.num_detected)
        return symbol
    def get_nu_symbol(self):
        # symbol = V(self.parent.SYMBOL_GAP * self.idx + self.num_detected)
        symbol = self.get_derivative_symbol(1)
        return symbol

    def get_derivative_symbol(self, order) -> int:
        return L(self.parent.SYMBOL_GAP * self.idx + self.num_detected + (self.parent.SYMBOL_GAP//self.parent.DERIVATIVE_SYMBOL_GAP) * order)

    def get_T_wo_init(self) -> gtsam.Pose3:
        attribute_name = "T_wo_init"
        if self.cached_attributes[attribute_name][0] < self.parent.update_stamp:
            self.cached_attributes[attribute_name][0] = self.parent.update_stamp
            self.cached_attributes[attribute_name][1] = self.parent.factor_graph.current_estimate.atPose3(self.get_symbol())
        return self.cached_attributes[attribute_name][1]

    def get_Q_init(self) -> gtsam.Pose3:
        attribute_name = "Q_init"
        if self.cached_attributes[attribute_name][0] < self.parent.update_stamp:
            self.cached_attributes[attribute_name][0] = self.parent.update_stamp
            self.cached_attributes[attribute_name][1] = self.parent.factor_graph.marginalCovariance(self.get_symbol())
        return self.cached_attributes[attribute_name][1]

    def get_nu_init(self):
        attribute_name = "nu_init"
        if self.cached_attributes[attribute_name][0] < self.parent.update_stamp:
            self.cached_attributes[attribute_name][0] = self.parent.update_stamp
            if self.num_detected >= 2:
                self.cached_attributes[attribute_name][1] = self.parent.factor_graph.current_estimate.atVector(self.get_nu_symbol())
            else:
                self.cached_attributes[attribute_name][1] = np.zeros(6)
        return self.cached_attributes[attribute_name][1]

    def get_T_wo_extrapolated(self, time_stamp) -> gtsam.Pose3:
        T_wo_init = self.get_T_wo_init()
        nu = self.get_nu_init()
        dt = time_stamp - self.last_seen_time_stamp
        T_wo = custom_odom_factors.plus_so3r3_global(T_wo_init, nu, dt)
        return T_wo

    def get_velocity_Q(self, dt):
        Q = np.eye(6)
        Q[:3, :3] = np.eye(3) * self.parent.params.cov_drift_ang_vel * dt
        Q[3:6, 3:6] = np.eye(3) * self.parent.params.cov_drift_lin_vel * dt
        return Q

    def get_Q_extrapolated(self, time_stamp):
        Q_init = self.get_Q_init()
        dt = time_stamp - self.last_seen_time_stamp
        Q = Q_init + self.get_velocity_Q(dt)
        return Q

    def __repr__(self):
        # return f"obj_label:{self.obj_label}, idx:{self.idx}, num_detected:{self.num_detected}, last_seen_time_stamp:{self.last_seen_time_stamp}"
        return f"(Track:{self.obj_label}-{self.idx})"

    def __hash__(self):
        return self.idx

    def add_detection(self, T_wc:gtsam.Pose3, T_co:gtsam.Pose3, Q:np.ndarray, time_stamp:float):
        T_co = gtsam.Pose3(T_co)
        T_wc = gtsam.Pose3(T_wc)
        # T_wc = self.parent.camera.get_T_wc_init()
        T_wo = T_wc * T_co
        self.num_detected += 1

        noise = gtsam.noiseModel.Gaussian.Covariance(Q)
        factor = gtsam.BetweenFactorPose3(self.parent.camera.get_symbol(), self.get_symbol(), T_co, noise)
        self.parent.factor_graph.add_factor(factor)
        self.parent.factor_graph.inser_estimate(self.get_symbol(), T_wo)
        for order in range(1, self.max_order_derivative + 1):  # order=1...velocity, order=2...acceleration, order=3...jerk ...
            if self.num_detected > order:
                dt = time_stamp - self.last_seen_time_stamp
                cov2 = np.eye(6)
                cov2[:3, :3] = np.eye(3) * self.parent.params.cov2_R
                cov2[3:6, 3:6] = np.eye(3) * self.parent.params.cov2_t
                triple_factor_noise = gtsam.noiseModel.Gaussian.Covariance(cov2)
                if order == 1:
                    error_func = partial(custom_odom_factors.error_velocity_integration_so3r3_global, dt)
                elif order > 1:
                    pass
                    #  TODO: this
                    # error_func = partial(custom_odom_factors.error_velocity_integration_so3r3_global, dt)
                factor = gtsam.CustomFactor(triple_factor_noise,[self.get_derivative_symbol(order-1) - 1,
                                                                 self.get_derivative_symbol(order-1),
                                                                 self.get_derivative_symbol(order)],
                                                                 error_func)
                self.parent.factor_graph.add_factor(factor)
                self.parent.factor_graph.inser_estimate(self.get_derivative_symbol(order), np.zeros((6)))
        if self.num_detected > self.max_order_derivative:
            between_noise = gtsam.noiseModel.Gaussian.Covariance(self.get_velocity_Q(dt))
            self.parent.factor_graph.add_factor(gtsam.BetweenFactorVector(self.get_derivative_symbol(self.max_order_derivative) - 1,
                                                                          self.get_derivative_symbol(self.max_order_derivative),
                                                                          np.zeros(6),
                                                                          between_noise))
        # if self.num_detected >= 2:
        #     dt = time_stamp - self.last_seen_time_stamp
        #     error_func = partial(custom_odom_factors.error_velocity_integration_so3r3_global, dt)
        #     cov2 = np.eye(6)
        #     cov2[:3, :3] = np.eye(3) * self.parent.params.cov2_R * dt
        #     cov2[3:6, 3:6] = np.eye(3) * self.parent.params.cov2_t * dt
        #     prior_int_twist = gtsam.noiseModel.Gaussian.Covariance(cov2)
        #     factor = gtsam.CustomFactor(prior_int_twist, [self.get_symbol() - 1, self.get_symbol(), self.get_nu_symbol()], error_func)
        #     self.parent.factor_graph.add_factor(factor)
        #     velocity_estimate = gtsam.Pose3.Logmap(self.get_T_wo_init().inverse() * T_wo)
        #     # self.parent.factor_graph.inser_estimate(self.get_nu_symbol(), np.zeros(6))  # TODO: use a better estimate
        #     self.parent.factor_graph.inser_estimate(self.get_nu_symbol(), velocity_estimate)  # TODO: use a better estimate
        # if self.num_detected >= 3:
        #     velocity_noise = gtsam.noiseModel.Gaussian.Covariance(self.get_velocity_Q(dt))
        #     self.parent.factor_graph.add_factor(gtsam.BetweenFactorVector(self.get_nu_symbol() - 1, self.get_nu_symbol(), np.zeros(6), velocity_noise))
        self.last_seen_time_stamp = time_stamp
        self.parent.last_time_stamp = time_stamp


class Tracks:
    def __init__(self, params:GlobalParams):
        self.params = params
        self.factor_graph = FactorGraphWrapper(params)
        self.tracks: defaultdict[str:[Symbol]] = defaultdict(set)
        self.created_tracks = 0
        self.camera:Camera = Camera(self)
        self.update_stamp = 0
        self.last_time_stamp = None

        self.SYMBOL_GAP = 10 ** 8
        self.DERIVATIVE_SYMBOL_GAP = 10**2  #  the highest order of a derivative that can be used for a track


    def remove_expired_tracks(self, current_time_stamp):
        for obj_label in self.tracks:
            for track in list(self.tracks[obj_label]):
                if (current_time_stamp - track.last_seen_time_stamp) > self.params.max_track_age:
                    self.tracks[obj_label].remove(track)
                    del track


    def create_track(self, obj_label):
        new_track = Track(self, self.created_tracks + 1, obj_label)
        self.created_tracks += 1
        self.tracks[obj_label].add(new_track)
        return new_track

    def calculate_D(self, tracks, detections):
        """
        Calculates a 2d matrix containing distances between each new and old object estimates
        """
        D = np.full((len(tracks), len(detections)), np.inf, dtype=float)
        T_wc:gtsam.Pose3 = self.camera.get_T_wc_init()
        for i in range(len(tracks)):
            track: Track = tracks[i]
            T_co_track:gtsam.Pose3 = T_wc.inverse() * track.get_T_wo_init()
            # Q_wo_track =
            for j in range(len(detections)):
                T_co_detection = gtsam.Pose3(detections[j]["T_co"])
                Q_co_detection = detections[j]["Q"]
                T_oo = T_co_track.inverse() * T_co_detection
                w = gtsam.Pose3.Logmap(T_oo.inverse())
                D[i, j] = mahalanobis_distance(w, Q_co_detection)
        return D

    def get_tracks_matches(self, obj_label, detections):
        assignment = [None for i in range(len(detections))]
        tracks = list(self.tracks[obj_label])
        D = self.calculate_D(tracks, detections)
        for i in range(min(D.shape[0], D.shape[1])):
            arg_min = np.unravel_index(np.argmin(D, axis=None), D.shape)
            minimum = D[arg_min]
            D[arg_min[0], :] = np.full((D.shape[1]), np.inf)
            D[:, arg_min[1]] = np.full((D.shape[0]), np.inf)
            if minimum < self.params.outlier_rejection_treshold:
                assignment[i] = tracks[arg_min[0]]
        return assignment
