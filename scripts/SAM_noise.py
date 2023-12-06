import gtsam
from gtsam import Symbol
from gtsam.symbol_shorthand import B, V, X
from typing import List, Dict
import numpy as np
class SAM_noise():


    @staticmethod
    def transform_cov(T_ba:np.ndarray, C_ac:np.ndarray):
        assert C_ac.shape == (6, 6)
        if isinstance(T_ba, gtsam.Pose3):
            T_ba = T_ba.matrix()
        assert T_ba.shape == (4, 4) or T_ba.shape == (3, 3)
        C_bc = C_ac
        C_bc[:3, :3] = T_ba[:3, :3] @ C_bc[:3, :3] @ T_ba[:3, :3].T
        C_bc[3:6, 3:6] = T_ba[:3, :3] @ C_bc[3:6, 3:6] @ T_ba[:3, :3].T
        return C_bc

    @staticmethod
    def get_panda_eef_noise() -> gtsam.noiseModel:
        exp_sigma = 0.0001  # norm in radians
        xyz_sigma = 0.0001  # in meters
        # exp, x, y, z
        C_cc = np.diag([exp_sigma,
                               exp_sigma,
                               exp_sigma,
                               xyz_sigma,
                               xyz_sigma,
                               xyz_sigma])
        noise = gtsam.noiseModel.Gaussian.Covariance(C_cc)
        return noise

    @staticmethod
    def get_object_in_camera_noise(T_co: np.ndarray, T_bc:np.ndarray,  f: float = 0.03455) -> gtsam.noiseModel:
        """
        :param f: camera focal length in meters
        """
        exp_sigma = 0.01  # norm in radians
        xyz_sigma = 0.000005  # in meters
        z = T_co[2, 3]  # distance from camera
        a = 0.05  # object size estimate, meters
        C_co = np.diag([exp_sigma,
                               exp_sigma,
                               exp_sigma,
                               (xyz_sigma * z)/f,
                               (xyz_sigma * z)/f,
                               (xyz_sigma * z**2)/(a*f*f)])
        T_oc = np.linalg.inv(T_co)
        C_oo = SAM_noise.transform_cov(T_oc, C_co)
        # cov[3:6, 3:6] = T_bc[:3, :3] @ cov[3:6, 3:6] @ T_bc[:3, :3].T  # transform covariance matrix from camera frame to object frame
        noise = gtsam.noiseModel.Gaussian.Covariance(C_oo)
        return noise


def main():
    pass

if __name__ == "__main__":
    main()

