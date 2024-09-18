from dataclasses import dataclass
@dataclass
class GlobalParams:
    # translation_dist_weight: float = 1.0
    mod: int = 1
    cov_drift_lin_vel: float = 0.001
    cov_drift_ang_vel: float = 0.1
    cov2_t: float = 0.000000000001
    cov2_R: float = 0.000000000001
    t_validity_treshold: float = 2.5e-5
    R_validity_treshold: float = 2e2
    max_track_age: float = 1.0  # in seconds
    chunk_size: int = 20
    outlier_rejection_treshold_trans: float = 10
    outlier_rejection_treshold_rot: float = 10
    velocity_prior_sigma: float = 10
    # velocity_diminishing_coef: float = 0.99
    # hysteresis_coef: float = 1
    max_derivative_order: int = 2  # 1...const velocity, 2...const acceleration, ...
    reject_overlaps:float = 0  # in meters, if 2 detections of the same object are closer than this value, than only one is deemed valid

    def __repr__(self):
        return f"{self.mod}_" \
               f"{self.max_derivative_order}_"\
               f"{self.max_track_age}_" \
               f"{self.cov_drift_lin_vel}_" \
               f"{self.cov_drift_ang_vel}_" \
               f"{self.outlier_rejection_treshold_trans}_" \
               f"{self.outlier_rejection_treshold_rot}_" \
               f"{self.t_validity_treshold:.2E}_" \
               f"{self.R_validity_treshold:.2E}"


rapid_tracking = GlobalParams(mod=1,
                                chunk_size=20,
                                cov_drift_lin_vel=0.4,
                                cov_drift_ang_vel=20.0,
                                max_track_age=1.0,
                                outlier_rejection_treshold_trans=0.15,
                                outlier_rejection_treshold_rot=0.25,
                                t_validity_treshold=0.01,
                                R_validity_treshold=1.0,
                                velocity_prior_sigma=10,
                                max_derivative_order=2)

robust_tracking = GlobalParams(
                                cov_drift_lin_vel=0.0004,
                                cov_drift_ang_vel=0.04,
                                outlier_rejection_treshold_trans=0.15,
                                outlier_rejection_treshold_rot=0.25,
                                t_validity_treshold=3.2e-4,
                                R_validity_treshold=0.16,
                                max_track_age=3.0,
                                # t_validity_treshold=1e3,
                                # R_validity_treshold=1e3,
                                max_derivative_order=1)