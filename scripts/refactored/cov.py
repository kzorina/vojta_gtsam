#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2024-02-19
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

import numpy as np
from numpy.typing import ArrayLike
import pinocchio as pin


def normalized(v: ArrayLike) -> np.ndarray:
    """Normalize vector to unit length."""
    v = np.asarray(v)
    return v / np.sqrt(np.sum(v**2))


def rotation_that_transforms_a_to_b(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Find rotation that transform z_start to z_goal, i.e. Rcc_prime."""
    a = normalized(a)
    b = normalized(b)
    v = np.cross(a, b)
    ang = np.arccos(np.dot(a, b))
    return pin.exp3(ang * normalized(v))


def exponential_function(x, a: float, b: float) -> np.ndarray:
    return a * np.exp(-b * x)


def measurement_covariance(
    Tco: np.ndarray, pixel_size: int
) -> np.ndarray:
    """Covariance of the measurement."""
    # params for bbox_size:
    # params_xy = np.asarray([3.0e-03, 2.0e-05])
    # params_z = np.asarray([2.0e-02, 1.0e-05])
    # params_angle = np.asarray([1.0e-01, 3.0e-05])

    # params for px_count:
    params_xy = np.asarray([2.6e-03, 2.0e-05])
    params_z = np.asarray([2.2e-02, 1.5e-05])
    params_angle = np.asarray([1.4e-01, 3.7e-05])

    var_xy = exponential_function(pixel_size, *params_xy) ** 2
    var_z = exponential_function(pixel_size, *params_z) ** 2
    var_angle = exponential_function(pixel_size, *params_angle) ** 2

    cov_trans_cam_aligned = np.diag([var_xy, var_xy, var_z])
    rot = rotation_that_transforms_a_to_b([0, 0, 1], Tco[:3, 3])
    cov_trans_c = rot @ cov_trans_cam_aligned @ rot.T  # cov[AZ] = A cov[Z] A^T
    rot = Tco[:3, :3].T
    cov_trans_o = rot @ cov_trans_c @ rot.T  # cov[AZ] = A cov[Z] A^T

    cov_o = np.zeros((6, 6))
    cov_o[:3, :3] = np.diag([var_angle] * 3)
    cov_o[3:6, 3:6] = cov_trans_o
    return cov_o


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    T_co = np.eye(4)
    T_co[:3, 3] = [0.2, 0.2, 0.10]
    T_co[:3, :3] = pin.exp3(np.random.rand(3))

    cov = measurement_covariance(T_co, 1000)
    r = np.random.multivariate_normal(np.zeros(6), cov, size=(100,))

    sampled_poses = []
    for v in r:
        T_sampled = np.eye(4)
        T_sampled[:3, :3] = pin.exp3(v[:3])
        T_sampled[:3, 3] = v[3:]
        sampled_poses.append(T_co @ T_sampled)
    sampled_poses = np.asarray(sampled_poses)

    fig, ax = plt.subplots(1, 1, squeeze=True)  # type: plt.Figure, plt.Axes
    ax.plot(sampled_poses[:, 0, 3], sampled_poses[:, 2, 3], "o")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_aspect("equal")
    ax.set_xlim(-0.25, 0.25)
    ax.set_ylim(-0.25, 0.25)

    plt.show()
