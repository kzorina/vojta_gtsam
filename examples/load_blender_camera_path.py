"""
GTSAM Copyright 2010-2018, Georgia Tech Research Corporation,
Atlanta, Georgia 30332-0415
All Rights Reserved
Authors: Frank Dellaert, et al. (see THANKS for the full author list)

See LICENSE for the license information

Pose SLAM example using iSAM2 in 3D space.
Author: Jerred Chen
Modeled after:
    - VisualISAM2Example by: Duy-Nguyen Ta (C++), Frank Dellaert (Python)
    - Pose2SLAMExample by: Alex Cunningham (C++), Kevin Deng & Frank Dellaert (Python)
"""

from typing import List

import matplotlib.pyplot as plt
import numpy as np

import gtsam
from gtsam import Symbol
from gtsam.symbol_shorthand import B, V, X
import gtsam.utils.plot as gtsam_plot
from Vizualize import draw_3d_estimate
import pickle

def create_poses() -> List[gtsam.Pose3]:
    """Creates ground truth poses of the robot."""
    P0 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    P1 = np.array([[0, -1, 0, 15],
                   [1, 0, 0, 15],
                   [0, 0, 1, 20],
                   [0, 0, 0, 1]])
    P2 = np.array([[np.cos(np.pi/4), 0, np.sin(np.pi/4), 30],
                   [0, 1, 0, 30],
                   [-np.sin(np.pi/4), 0, np.cos(np.pi/4), 30],
                   [0, 0, 0, 1]])
    P3 = np.array([[0, 1, 0, 30],
                   [0, 0, -1, 0],
                   [-1, 0, 0, -15],
                   [0, 0, 0, 1]])
    P4 = np.array([[-1, 0, 0, 0],
                   [0, -1, 0, -10],
                   [0, 0, 1, -10],
                   [0, 0, 0, 1]])
    P5 = P0[:]

    return [gtsam.Pose3(P0), gtsam.Pose3(P1), gtsam.Pose3(P2),
            gtsam.Pose3(P3), gtsam.Pose3(P4), gtsam.Pose3(P5)]

def load_poses(path) -> List[gtsam.Pose3]:
    """Creates ground truth poses of the robot."""
    dbfile = open(path, 'rb')
    poses = pickle.load(dbfile)
    dbfile.close()
    ret = []
    for entry in poses:
        P = entry["Camera"]
        ret.append(gtsam.Pose3(P))
    return ret

def load_statis_landmarks(path) -> List[gtsam.Pose3]:
    dbfile = open(path, 'rb')
    poses = pickle.load(dbfile)
    dbfile.close()
    ret = []
    for key in poses[0]:
        if key != "Camera":
            P = poses[0][key]
            ret.append(gtsam.Pose3(P))
    return ret

def determine_loop_closure(odom_tf: gtsam.Pose3, current_estimate: gtsam.Values,
    key: int, xyz_tol=0.6, rot_tol=17) -> int:
    """Simple brute force approach which iterates through previous states
    and checks for loop closure.

    Args:
        odom_tf: The noisy odometry transformation measurement in the body frame.
        current_estimate: The current estimates computed by iSAM2.
        key: Key corresponding to the current state estimate of the robot.
        xyz_tol: Optional argument for the translational tolerance, in meters.
        rot_tol: Optional argument for the rotational tolerance, in degrees.
    Returns:
        k: The key of the state which is helping add the loop closure constraint.
            If loop closure is not found, then None is returned.
    """
    if current_estimate:
        prev_est = current_estimate.atPose3(key+1)
        curr_est = prev_est.compose(odom_tf)
        for k in range(1, key+1):
            pose = current_estimate.atPose3(k)
            if (abs(pose.matrix()[:3,:3] - curr_est.matrix()[:3,:3]) <= rot_tol*np.pi/180).all() and \
                (abs(pose.matrix()[:3,3] - curr_est.matrix()[:3,3]) <= xyz_tol).all():
                    return k

def add_noise_to_measurement(measurement, noise):
    odometry_xyz = (measurement.x(), measurement.y(), measurement.z())
    odometry_rpy = measurement.rotation().rpy()
    noisy_measurements = np.random.multivariate_normal(np.hstack((odometry_rpy,odometry_xyz)), noise.covariance())
    return(gtsam.Pose3(gtsam.Rot3.RzRyRx(noisy_measurements[:3]), noisy_measurements[3:6].reshape(-1,1)))

def Pose3_ISAM2_example():
    """Perform 3D SLAM given ground truth poses as well as simple
    loop closure detection."""
    plt.ion()

    # Declare the 3D translational standard deviations of the prior factor's Gaussian model, in meters.
    prior_xyz_sigma = 0.03

    # Declare the 3D rotational standard deviations of the prior factor's Gaussian model, in degrees.
    prior_rpy_sigma = 0.5

    # Declare the 3D translational standard deviations of the odometry factor's Gaussian model, in meters.
    odometry_xyz_sigma = 0.01

    # Declare the 3D rotational standard deviations of the odometry factor's Gaussian model, in degrees.
    odometry_rpy_sigma = 0.1

    # Although this example only uses linear measurements and Gaussian noise models, it is important
    # to note that iSAM2 can be utilized to its full potential during nonlinear optimization. This example
    # simply showcases how iSAM2 may be applied to a Pose2 SLAM problem.
    PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([prior_rpy_sigma*np.pi/180,
                                                                prior_rpy_sigma*np.pi/180,
                                                                prior_rpy_sigma*np.pi/180,
                                                                prior_xyz_sigma,
                                                                prior_xyz_sigma,
                                                                prior_xyz_sigma]))
    ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([odometry_rpy_sigma*np.pi/180,
                                                                odometry_rpy_sigma*np.pi/180,
                                                                odometry_rpy_sigma*np.pi/180,
                                                                odometry_xyz_sigma,
                                                                odometry_xyz_sigma,
                                                                odometry_xyz_sigma]))

    # Create a Nonlinear factor graph as well as the data structure to hold state estimates.
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    # Create iSAM2 parameters which can adjust the threshold necessary to force relinearization and how many
    # update calls are required to perform the relinearization.
    parameters = gtsam.ISAM2Params()
    parameters.setRelinearizeThreshold(0.1)
    parameters.relinearizeSkip = 1
    isam = gtsam.ISAM2(parameters)

    # Create the ground truth poses of the robot trajectory.
    # true_poses = create_poses()
    true_poses = load_poses("/home/vojta/PycharmProjects/gtsam_playground/blender_dataset/bagr.p")
    true_landmarks = load_statis_landmarks("/home/vojta/PycharmProjects/gtsam_playground/blender_dataset/bagr.p")

    # Create the ground truth odometry transformations, xyz translations, and roll-pitch-yaw rotations
    # between each robot pose in the trajectory.

    # Add the prior factor to the factor graph, and poorly initialize the prior pose to demonstrate
    # iSAM2 incremental optimization.
    gt_dict = {}

    graph.add(gtsam.PriorFactorPose3(X(1), true_poses[0], PRIOR_NOISE))
    gt_dict[str(Symbol(X(1)).string())] = true_poses[0]

    initial_estimate.insert(X(1), true_poses[0])

    # Initialize the current estimate which is used during the incremental inference loop.
    current_estimate = initial_estimate
    detected_landmarks = set()



    for i in range(1, len(true_poses)):

        # Obtain the noisy translation and rotation that is received by the robot and corrupted by gaussian noise.
        odometry_tf = true_poses[i - 1].transformPoseTo(true_poses[i])
        noisy_tf = add_noise_to_measurement(odometry_tf, ODOMETRY_NOISE)

        previous_key = X(i)
        current_key = X(i + 1)

        graph.add(gtsam.BetweenFactorPose3(previous_key, current_key, noisy_tf, ODOMETRY_NOISE))
        gt_dict[str(Symbol(current_key).string())] = true_poses[i]
        noisy_estimate = current_estimate.atPose3(previous_key).compose(noisy_tf)
        initial_estimate.insert(current_key, noisy_estimate)

        for l in range(len(true_landmarks)):
            landmark_key = V(l)

            landmark_tf = true_poses[i - 1].transformPoseTo(true_landmarks[l])
            noisy_landmark_tf = add_noise_to_measurement(landmark_tf, ODOMETRY_NOISE)

            graph.add(gtsam.BetweenFactorPose3(previous_key, landmark_key, noisy_landmark_tf, ODOMETRY_NOISE))
            gt_dict[str(Symbol(landmark_key).string())] = true_landmarks[l]
            if landmark_key not in detected_landmarks:
                detected_landmarks.add(landmark_key)
                noisy_estimate = current_estimate.atPose3(previous_key).compose(noisy_landmark_tf)
                initial_estimate.insert(landmark_key, noisy_estimate)

        # Compute and insert the initialization estimate for the current pose using a noisy odometry measurement.
        # Perform incremental update to iSAM2's internal Bayes tree, optimizing only the affected variables.
        isam.update(graph, initial_estimate)
        current_estimate = isam.calculateEstimate()

        # Report all current state estimates from the iSAM2 optimization.
        # print(gt_dict)
        draw_3d_estimate(graph, current_estimate)

        initial_estimate.clear()

    # Print the final covariance matrix for each pose after completing inference.
    marginals = gtsam.Marginals(graph, current_estimate)
    i = 1
    while current_estimate.exists(i):
        print(f"X{i} covariance:\n{marginals.marginalCovariance(i)}\n")
        i += 1

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    Pose3_ISAM2_example()