#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2024-02-19
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

import numpy as np
from movement_primitives.dmp import CartesianDMP
import pinocchio as pin
import matplotlib.pyplot as plt


def sample_trajectory(number_of_waypoints=10, poses_per_waypoint=10):
    """Sample a trajectory with random waypoints and random DMPs.
    Return array of vectors representing pose with XYZQuat."""
    waypoints = [pin.SE3.Random() for _ in range(number_of_waypoints)]
    waypoints_velocities = [np.random.rand(6) - 0.5 for _ in range(number_of_waypoints)]
    waypoints_accelerations = [
        np.random.rand(6) - 0.5 for _ in range(number_of_waypoints)
    ]
    poses = []
    dmp = CartesianDMP(dt=1 / (poses_per_waypoint * 10))
    dmp.set_weights(1000 * (np.random.rand(*dmp.get_weights().shape) - 0.5))
    for i in range(1, number_of_waypoints):
        dmp.configure(
            start_y=pin.SE3ToXYZQUAT(waypoints[i - 1]),
            start_yd=waypoints_velocities[i - 1],
            start_ydd=waypoints_accelerations[i - 1],
            goal_y=pin.SE3ToXYZQUAT(waypoints[i]),
            goal_yd=waypoints_velocities[i],
            goal_ydd=waypoints_accelerations[i],
        )
        poses += dmp.open_loop(run_t=1.0)[1].tolist()[::10]
    return np.asarray(poses)


def plot_poses(poses: np.ndarray, ax: plt.Axes, scale=0.1):
    ax.plot(poses[:, 0], poses[:, 1], poses[:, 2], "k")
    for pose in poses:
        t = pin.XYZQUATToSE3(pose)
        o = t * np.array([0, 0, 0])
        x = t * np.array([scale, 0, 0])
        y = t * np.array([0, scale, 0])
        z = t * np.array([0, 0, scale])
        ax.plot([o[0], x[0]], [o[1], x[1]], [o[2], x[2]], "r")
        ax.plot([o[0], y[0]], [o[1], y[1]], [o[2], y[2]], "g")
        ax.plot([o[0], z[0]], [o[1], z[1]], [o[2], z[2]], "b")


fig: plt.Figure = plt.figure()
ax: plt.Axes = fig.add_subplot(111, projection="3d")
for i in range(10):
    plot_poses(sample_trajectory(3, 30), ax)
    fig.savefig(f"sample_trajectory_{i}.png")
    ax.clear()
plt.show()

