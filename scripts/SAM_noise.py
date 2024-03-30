import gtsam
import pinocchio as pin
from gtsam import Symbol
from gtsam.symbol_shorthand import B, V, X
from typing import List, Dict
from cov_model import compute_covariance
import numpy as np
from cov import measurement_covariance
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
    def get_panda_eef_cov():
        # exp_sigma = 0.000001  # norm in radians
        # xyz_sigma = 0.000001  # in meters
        exp_sigma = 0.000000001  # norm in radians
        xyz_sigma = 0.000000001  # in meters
        # exp, x, y, z
        C_cc = np.diag([exp_sigma,
                               exp_sigma,
                               exp_sigma,
                               xyz_sigma,
                               xyz_sigma,
                               xyz_sigma])
        return C_cc
    @staticmethod
    def get_panda_eef_noise() -> gtsam.noiseModel:
        C_cc = SAM_noise.get_panda_eef_cov()
        noise = gtsam.noiseModel.Gaussian.Covariance(C_cc)
        return noise


    @staticmethod
    def get_object_in_camera_noise_old(T_co: np.ndarray, T_bc:np.ndarray,  f: float = 0.03455) -> gtsam.noiseModel:
        """
        :param f: camera focal length in meters
        """
        exp_sigma = 0.002  # norm in radians
        xyz_sigma = 0.00001  # in meters
        z = T_co[2, 3]  # distance from camera
        a = 0.010  # object size estimate, meters
        v = np.array(((xyz_sigma * z)/f,
                      (xyz_sigma * z)/f,
                      (xyz_sigma * 0.05 * z**2)/(a*f*f)))
        C_co = np.diag([exp_sigma, exp_sigma, exp_sigma, 0, 0, 0])
        C_co[3:6, 3:6] = np.diag(v)
        T_oc = np.linalg.inv(T_co)
        C_oo = SAM_noise.transform_cov(T_oc, C_co)
        # cov[3:6, 3:6] = T_bc[:3, :3] @ cov[3:6, 3:6] @ T_bc[:3, :3].T  # transform covariance matrix from camera frame to object frame
        noise = gtsam.noiseModel.Gaussian.Covariance(C_oo)
        return noise

    @staticmethod
    def get_object_in_camera_noise(T_co: np.ndarray, T_bc:np.ndarray,  f: float = 0.03455) -> gtsam.noiseModel:
        """
        :param f: camera focal length in meters
        """
        exp_sigma = 0.002  # norm in radians
        xyz_sigma = 0.00001  # in meters
        exp_sigma = 0.004  # norm in radians
        exp_sigma = 0.0001  # norm in radians
        xyz_sigma = 0.00001  # in meters
        z = T_co[2, 3]  # distance from camera
        a = 0.010  # object size estimate, meters
        v = np.array(((xyz_sigma * z)/f,
                      (xyz_sigma * z)/f,
                      (xyz_sigma * 2 * z**2)/(a*f)))
        dir = T_co[:3, 3]/np.linalg.norm(T_co[:3, 3])
        C_co = np.diag([exp_sigma, exp_sigma, exp_sigma, 0, 0, 0])
        w = np.cross(np.array([0, 0, 1]), dir)
        w = (w/np.linalg.norm(w)) * np.arccos(np.dot(dir, np.array([0, 0, 1])))
        R = pin.exp3(w)
        C_co[3:6, 3:6] = R @ np.diag(v) @ R.T
        # T_oc = np.linalg.inv(T_co)
        T_oc = gtsam.Pose3(T_co).inverse().matrix()
        C_oo = SAM_noise.transform_cov(T_oc, C_co)
        # cov[3:6, 3:6] = T_bc[:3, :3] @ cov[3:6, 3:6] @ T_bc[:3, :3].T  # transform covariance matrix from camera frame to object frame
        noise = gtsam.noiseModel.Gaussian.Covariance(C_oo)
        return noise

    @staticmethod
    def get_object_in_camera_noise_px(T_co: np.ndarray, px_count) -> gtsam.noiseModel:
        """
        :param f: camera focal length in meters
        """
        C_oo = measurement_covariance(T_co, px_count)
        C_oo = C_oo
        #  TODO:change this back ASAP
        noise = gtsam.noiseModel.Gaussian.Covariance(C_oo)
        return noise


    @staticmethod
    def get_object_in_camera_noise2(T_co: np.ndarray, T_wc:np.ndarray,  K: np.ndarray) -> gtsam.noiseModel:
        """
        :param f: camera focal length in meters
        """
        sig = 10000
        tag_width = 0.3
        sig_pix = 1
        a_corners = 0.5 * tag_width * np.array([
            [-1, 1, 0],  # bottom left
            [1, 1, 0],  # bottom right
            [1, -1, 0],  # top right
            [-1, -1, 0],  # top left
        ])
        Q_co_pin = compute_covariance(T_co[:3, 3], T_co[:3, :3], K, a_corners, sig_pix)
        Q_co = np.roll(Q_co_pin, shift=(3, 3), axis=(0, 1))
        Q_oo = SAM_noise.transform_cov(T_co, Q_co)
        Q_oo = Q_oo*sig
        noise = gtsam.noiseModel.Gaussian.Covariance(Q_oo)
        return noise


def main():
    pass

if __name__ == "__main__":
    main()

