from typing import List, Dict, Set
import gtsam
import numpy as np
from gtsam import Symbol
from gtsam.symbol_shorthand import B, V, X, L
from FactorGraphWrapper import FactorGraphWrapper
from GlobalParams import GlobalParams
from distribution_distances import mahalanobis_distance, bhattacharyya_distance, euclidean_distance, translation_distance, rotation_distance
from collections import defaultdict
import custom_odom_factors
from functools import partial
from scipy.special import factorial
from State import BareTrack
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
        self.parent.factor_graph.insert_estimate(self.get_symbol(), T_wc)
        self.cached_attributes["T_wo_init"][0] = self.parent.update_stamp
        self.cached_attributes["T_wo_init"][1] = T_wc
        self.cached_attributes["Q_init"][0] = self.parent.update_stamp
        self.cached_attributes["Q_init"][1] = Q

    def get_T_wc_init(self) -> gtsam.Pose3:
        attribute_name = "T_wo_init"
        if self.cached_attributes[attribute_name][0] < self.parent.update_stamp:
            self.cached_attributes[attribute_name][0] = self.parent.update_stamp
            self.cached_attributes[attribute_name][1] = None
        return self.cached_attributes[attribute_name][1]

    def get_Q_init(self) -> gtsam.Pose3:
        attribute_name = "Q_init"
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
        # self.max_order_derivative = 2  # 1...const velocity, 2...const acceleration, ...
        self.max_order_derivative = parent.params.max_derivative_order
        self.cached_derivatives = [[-1, None] for i in range(self.max_order_derivative + 2)]
        self.cached_Q_derivatives = [[-1, None] for i in range(self.max_order_derivative + 2)]

    @staticmethod
    def extrapolate_func(dt:float, T_wo:gtsam.Pose3, derivatives:np.ndarray, Q:np.ndarray, Q_derivatives:np.ndarray):
        order = derivatives.shape[0]

        dts = np.full((order), dt)
        i = np.arange((order)) + 1
        ret = (dts**i)/factorial(i)
        x = ret[np.newaxis].T

        tau = np.sum(derivatives * x, axis=0)
        new_Q = Q + np.sum(Q_derivatives * x[:, np.newaxis], axis=0)
        R_new = gtsam.Rot3.Expmap(tau[:3]) * T_wo.rotation()
        t_new = T_wo.translation() + tau[3:6]
        new_T_wo = gtsam.Pose3(R_new, t_new)
        return new_T_wo, new_Q

    def get_active_derivative_symbols(self):
        ret = []
        for order in range(0, min(self.max_order_derivative, self.num_detected - 1)):
            ret.append(self.get_derivative_symbol(order + 1))
        return ret

    def get_symbol(self):
        symbol = L(self.parent.SYMBOL_GAP * self.idx + self.num_detected)
        return symbol

    def get_derivative_symbol(self, order) -> int:
        return L(self.parent.SYMBOL_GAP * self.idx + self.num_detected + (self.parent.SYMBOL_GAP//self.parent.DERIVATIVE_SYMBOL_GAP) * order)

    def get_T_wo_init(self) -> gtsam.Pose3:
        if self.cached_derivatives[0][0] < self.parent.update_stamp:
            self.cached_derivatives[0][0] = self.parent.update_stamp
            self.cached_derivatives[0][1] = self.parent.factor_graph.current_estimate.atPose3(self.get_symbol())
        return self.cached_derivatives[0][1]

    def get_derivative(self, order) -> np.ndarray:
        assert order > 0
        if order >= self.num_detected:
            return np.zeros((6))
        if order == self.max_order_derivative + 1:
            return np.zeros((6))
        if self.cached_derivatives[order][0] < self.parent.update_stamp:
            self.cached_derivatives[order][0] = self.parent.update_stamp
            self.cached_derivatives[order][1] = self.parent.factor_graph.current_estimate.atVector(self.get_derivative_symbol(order))
        return self.cached_derivatives[order][1]

    def get_Q_derivative(self, order) -> np.ndarray:  # in the world frame
        if order == self.max_order_derivative + 1:
            return self.get_highest_derivative_Q(1)
        if order >= self.num_detected:
            return np.eye(6)
        if self.cached_Q_derivatives[order][0] < self.parent.update_stamp:
            self.cached_Q_derivatives[order][0] = self.parent.update_stamp
            Q = self.parent.factor_graph.marginalCovariance(self.get_derivative_symbol(order))
            if order == 0:  # express in world frame
                T_wo = self.get_T_wo_init()
                R_wo = np.eye(4)
                R_wo[:3, :3] = T_wo.rotation().matrix()
                R_wo = gtsam.Pose3(R_wo)
                Q = R_wo.AdjointMap() @ Q @ R_wo.AdjointMap().T
            self.cached_Q_derivatives[order][1] = Q
        return self.cached_Q_derivatives[order][1]

    def get_bare_track(self):
        derivatives = np.zeros((self.max_order_derivative + 1, 6))
        Q_derivatives = np.zeros((self.max_order_derivative + 1, 6, 6))
        T_wo = self.get_T_wo_init()
        Q = self.get_Q_derivative(0)
        for i in range(1, self.max_order_derivative + 2):
            derivatives[i-1, :] = self.get_derivative(i)
            Q_derivatives[i-1, :, :] = self.get_Q_derivative(i)
        bare_track = BareTrack(self.idx, T_wo, Q, derivatives, Q_derivatives, self.last_seen_time_stamp, self.extrapolate_func)
        return bare_track

    def get_highest_derivative_Q(self, dt):
        Q = np.eye(6)
        Q[:3, :3] = np.eye(3) * self.parent.params.cov_drift_ang_vel * dt
        Q[3:6, 3:6] = np.eye(3) * self.parent.params.cov_drift_lin_vel * dt
        return Q

    # def get_Q_extrapolated(self, time_stamp):
    #     Q_init = self.get_Q_init()
    #     dt = time_stamp - self.last_seen_time_stamp
    #     Q = Q_init + self.get_velocity_Q(dt)
    #     return Q

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
        self.parent.factor_graph.insert_estimate(self.get_symbol(), T_wo)
        for order in range(1, self.max_order_derivative + 1):  # order=1...velocity, order=2...acceleration, order=3...jerk ...
            if self.num_detected > order:
                dt = time_stamp - self.last_seen_time_stamp
                cov2 = np.eye(6)
                cov2[:3, :3] = np.eye(3) * self.parent.params.cov2_R
                cov2[3:6, 3:6] = np.eye(3) * self.parent.params.cov2_t
                triple_factor_noise = gtsam.noiseModel.Gaussian.Covariance(cov2)
                if order == 1:  # "velocity - velocity- acceleration" triple factor
                    error_func = partial(custom_odom_factors.error_velocity_integration_so3r3_global, dt)
                else:  # Pose3 - Pose3 - velocity triple factor
                    error_func = partial(custom_odom_factors.error_derivative_integration_so3r3_global, dt)
                factor = gtsam.CustomFactor(triple_factor_noise,[self.get_derivative_symbol(order-1) - 1,
                                                                 self.get_derivative_symbol(order-1),
                                                                 self.get_derivative_symbol(order)],
                                                                 error_func)
                self.parent.factor_graph.add_factor(factor)
                self.parent.factor_graph.insert_estimate(self.get_derivative_symbol(order), np.zeros((6)))  # TODO: use better estimate
                if self.num_detected == order + 1:
                    bogus_noise = gtsam.noiseModel.Isotropic.Sigma(6, self.parent.params.velocity_prior_sigma)
                    self.parent.factor_graph.add_factor(gtsam.PriorFactorVector(self.get_derivative_symbol(order), np.zeros((6)), bogus_noise))

        if self.num_detected > self.max_order_derivative + 1:  # the highest order derivative between factor is ready to be added
            if self.max_order_derivative == 0:    # const pose model - the between factor is of type Pose3
                dt = time_stamp - self.last_seen_time_stamp
                between_noise = gtsam.noiseModel.Gaussian.Covariance(self.get_highest_derivative_Q(dt))
                self.parent.factor_graph.add_factor(gtsam.BetweenFactorPose3(self.get_derivative_symbol(self.max_order_derivative) - 1,
                                                                              self.get_derivative_symbol(self.max_order_derivative),
                                                                              gtsam.Pose3(np.eye(4)),
                                                                              between_noise))
            else:    # the between factor is of type Vector
                dt = time_stamp - self.last_seen_time_stamp
                between_noise = gtsam.noiseModel.Gaussian.Covariance(self.get_highest_derivative_Q(dt))
                self.parent.factor_graph.add_factor(gtsam.BetweenFactorVector(self.get_derivative_symbol(self.max_order_derivative) - 1,
                                                                              self.get_derivative_symbol(self.max_order_derivative),
                                                                              np.zeros(6),
                                                                              between_noise))

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

        self.SYMBOL_GAP = 10 ** 10
        self.DERIVATIVE_SYMBOL_GAP = 10**2  # the highest order of a derivative that can be used for a track


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

    def calculate_D(self, tracks, detections, time_stamp, distamce_type='m'):
        """
        Calculates a 2d matrix containing distances between each new and old object estimates
        """
        D = np.full((len(tracks), len(detections)), np.inf, dtype=float)
        T_wc:gtsam.Pose3 = self.camera.get_T_wc_init()
        Q_cc = self.camera.get_Q_init()
        for i in range(len(tracks)):
            track: Track = tracks[i]
            bare_track = track.get_bare_track()
            T_wo, Q_wo = bare_track.extrapolate(time_stamp)

            T_co_track:gtsam.Pose3 = T_wc.inverse() * T_wo
            # Q_wo_track =
            for j in range(len(detections)):
                T_co_detection = gtsam.Pose3(detections[j]["T_co"])
                Q_oo_detection = detections[j]["Q"]
                T_oo = T_co_track.inverse() * T_co_detection
                T_wo_detection = T_wc * T_co_detection
                R_wo_detection = np.eye(4)
                R_wo_detection[:3, :3] = T_wo_detection.rotation().matrix()
                R_wo_detection = gtsam.Pose3(R_wo_detection)
                Q_wo_detection = (R_wo_detection * T_co_detection.inverse()).AdjointMap() @ Q_cc @ (R_wo_detection * T_co_detection.inverse()).AdjointMap().T + R_wo_detection.AdjointMap() @ Q_oo_detection @ R_wo_detection.AdjointMap().T
                # W_wo_detection = gtsam.Pose3.Logmap(T_wo_detection)
                W_no = gtsam.Pose3.Logmap(T_co_detection.inverse() * T_co_track)
                W_ww = np.zeros(6)
                W_ww[:3] = gtsam.Rot3.Logmap(T_wo_detection.rotation().inverse() * T_wo.rotation())
                W_ww[3:6] = T_wo.translation() - T_wo_detection.translation()

                W_ww_detection = np.zeros(6)
                W_ww_detection[:3] = gtsam.Rot3.Logmap(T_wo.rotation().inverse() * T_wo_detection.rotation())
                W_ww_detection[3:6] = T_wo_detection.translation() - T_wo.translation()

                w = gtsam.Pose3.Logmap(T_wo_detection.inverse())
                if distamce_type == 'e':
                    D[i, j] = euclidean_distance(T_wo, T_wo_detection)
                if distamce_type == 'trans':
                    D[i, j] = translation_distance(T_wo, T_wo_detection)
                if distamce_type == 'rot':
                    D[i, j] = rotation_distance(T_wo, T_wo_detection)
                if distamce_type == 'mahal':
                    # D[i, j] = mahalanobis_distance(W_ww, Q_wo_detection)
                    D[i, j] = mahalanobis_distance(W_no, Q_oo_detection)
                if distamce_type == 'min_mahal':
                    D[i, j] = min(mahalanobis_distance(W_ww, Q_wo_detection), mahalanobis_distance(W_ww_detection, Q_wo))
                    # raise Exception(f"not implemented yet")
                # if distamce_type == 'b':
                #     D[i, j] = bhattacharyya_distance(W_wo, W_wo_detection, Q_wo, Q_wo_detection)
        return D

    def get_tracks_matches(self, obj_label, detections, time_stamp):
        assignment = [None for i in range(len(detections))]
        tracks = list(self.tracks[obj_label])
        D_match = self.calculate_D(tracks, detections, time_stamp, 'mahal')
        # D_outlier1 = self.calculate_D(tracks, detections, time_stamp, 'mahal')
        D_outlier_trans = self.calculate_D(tracks, detections, time_stamp, 'trans')
        D_outlier_rot = self.calculate_D(tracks, detections, time_stamp, 'rot')
        # print(f"D_match:{D_match}, D_outlier:{D_outlier}")
        for i in range(min(D_match.shape[0], D_match.shape[1])):
            arg_min = np.unravel_index(np.argmin(D_match, axis=None), D_match.shape)
            # minimum = D_match[arg_min]
            minimum_trans = D_outlier_trans[arg_min]
            minimum_rot = D_outlier_rot[arg_min]
            D_match[arg_min[0], :] = np.full((D_match.shape[1]), np.inf)
            D_match[:, arg_min[1]] = np.full((D_match.shape[0]), np.inf)
            if minimum_trans < self.params.outlier_rejection_treshold_trans and minimum_rot < self.params.outlier_rejection_treshold_rot:
                assignment[arg_min[1]] = tracks[arg_min[0]]
        # if obj_label == "Corn":
        #     D_match = self.calculate_D(tracks, detections, time_stamp, 'e')
        #     D_outlier = self.calculate_D(tracks, detections, time_stamp, 'e')
        #     print('')
        return assignment
