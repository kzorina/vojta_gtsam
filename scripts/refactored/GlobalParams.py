from dataclasses import dataclass
@dataclass
class GlobalParams:
    translation_dist_weight: float = 1.0
    mod: int = 1
    cov_drift_lin_vel: float = 0.01
    cov_drift_ang_vel: float = 0.01
    cov2_t: float = 0.000000001
    cov2_R: float = 0.000000001
    t_validity_treshold: float = 0.000005
    R_validity_treshold: float = 0.001
    max_track_age: int = 2.0  # in seconds
    chunk_size: int = 20
    outlier_rejection_treshold: float = 10
    velocity_prior_sigma: float = 10
    velocity_diminishing_coef: float = 0.99
    hysteresis_coef: float = 1

    def __repr__(self):
        return f"{self.mod}_" \
               f"{self.window_size}_" \
               f"{self.hysteresis_coef}_" \
               f"{self.cov_drift_lin_vel}_" \
               f"{self.cov_drift_ang_vel}_" \
               f"{self.cov2_t:.2E}_" \
               f"{self.cov2_R:.2E}_" \
               f"{self.outlier_rejection_treshold}_" \
               f"{self.t_validity_treshold:.2E}_" \
               f"{self.R_validity_treshold:.2E}"