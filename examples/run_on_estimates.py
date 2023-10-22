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
from pathlib import Path

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

def load_static_landmarks(path) -> List[gtsam.Pose3]:
    dbfile = open(path, 'rb')
    poses = pickle.load(dbfile)
    dbfile.close()
    ret = []
    for key in poses[0]:
        if key != "Camera":
            P = poses[0][key]
            ret.append(gtsam.Pose3(P))
    return ret

def load_landmark_measurements(path) -> List[dict[gtsam.Pose3]]:
    dbfile = open(path, 'rb')
    poses = pickle.load(dbfile)
    dbfile.close()
    ret = []
    for frame in poses:
        entry = {}
        for key in frame:
            if key != "Camera":
                P = poses[0][key]
                entry[key] = gtsam.Pose3(P)
        ret.append(entry)
    return ret

def add_noise_to_measurement(measurement, noise):
    odometry_xyz = (measurement.x(), measurement.y(), measurement.z())
    odometry_rpy = measurement.rotation().rpy()
    noisy_measurements = np.random.multivariate_normal(np.hstack((odometry_rpy,odometry_xyz)), noise.covariance())
    return(gtsam.Pose3(gtsam.Rot3.RzRyRx(noisy_measurements[:3]), noisy_measurements[3:6].reshape(-1,1)))

def add_noise_to_measurement_wrt_frame(measurement, noise, frame):
    odometry_xyz = (measurement.x(), measurement.y(), measurement.z())
    odometry_rpy = measurement.rotation().rpy()
    noisy_measurements = np.random.multivariate_normal(np.hstack((odometry_rpy,odometry_xyz)), noise.covariance())
    return(gtsam.Pose3(gtsam.Rot3.RzRyRx(noisy_measurements[:3]), noisy_measurements[3:6].reshape(-1,1)))


def estimate_to_dict(graph: gtsam.NonlinearFactorGraph, current_estimate: gtsam.Values, landmark_keys: dict, current_camera_key):
    inv_landmark_keys = {v: k for k, v in landmark_keys.items()}
    entry = {}
    T_wc = current_estimate.atPose3(current_camera_key)
    for key in graph.keyVector():
        if key in inv_landmark_keys:
            T_wo: gtsam.Pose3 = current_estimate.atPose3(key)
            T_co = T_wc.transformPoseTo(T_wo)
            object_name = inv_landmark_keys[key]
            entry[object_name] = T_co.matrix()


    return entry

def save_data(path, data):
    with open(path, 'wb') as file:
        pickle.dump(data, file)

def Pose3_ISAM2_example():
    """Perform 3D SLAM given ground truth poses as well as simple
    loop closure detection."""
    plt.ion()

    # Declare the 3D translational standard deviations of the prior factor's Gaussian model, in meters.
    prior_xyz_sigma = 0.00001

    # Declare the 3D rotational standard deviations of the prior factor's Gaussian model, in degrees.
    prior_rpy_sigma = 0.00001

    # Declare the 3D translational standard deviations of the odometry factor's Gaussian model, in meters.
    odometry_xyz_sigma = 0.00001

    # Declare the 3D rotational standard deviations of the odometry factor's Gaussian model, in degrees.
    odometry_rpy_sigma = 0.00001

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

    LANDMARK_NOISE:gtsam.noiseModel.Diagonal = gtsam.noiseModel.Diagonal.Sigmas(np.array([odometry_rpy_sigma*np.pi/180,
                                                                odometry_rpy_sigma*np.pi/180,
                                                                odometry_rpy_sigma*np.pi/180,
                                                                odometry_xyz_sigma/5,
                                                                odometry_xyz_sigma*2,
                                                                odometry_xyz_sigma/5]))
    # FALSE_LANDMARK_NOISE = gtsam.noiseModel.

    # PRIOR_NOISE = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
    #
    # ODOMETRY_NOISE = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
    #
    # LANDMARK_NOISE = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)

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
    dataset_path = Path(__file__).parent.parent / "datasets" / "crackers_new"
    true_poses = load_poses(dataset_path / "frames_gt.p")
    true_landmarks = load_static_landmarks(dataset_path / "frames_gt.p")
    landmark_measurements = load_landmark_measurements(dataset_path / "frames_prediction.p")

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
    landmark_keys = {}
    landmark_keys_idx = 0

    first_entry = {}
    T_wc = true_poses[0]
    for key in landmark_measurements[0]:
        T_co = landmark_measurements[0][key]
        first_entry[key] = T_co.matrix()
    estimate_frames = [first_entry]
    # estimate_frames = [landmark_measurements[0]]

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

        for dict_key in landmark_measurements[i - 1]:
            if dict_key not in landmark_keys:
                landmark_keys[dict_key] = V(landmark_keys_idx)
                landmark_keys_idx += 1
            landmark_key = landmark_keys[dict_key]

            landmark_tf = landmark_measurements[i - 1][dict_key]
            # landmark_tf = true_poses[i - 1].transformPoseTo(true_landmarks[l])

            noisy_landmark_tf = landmark_tf

            my_cov = np.array([[0.00001, 0, 0, 0, 0, 0],
                               [0, 0.00001, 0, 0, 0, 0],
                               [0, 0, 0.00001, 0, 0, 0],
                               [0, 0, 0, 0.00005, 0, 0],
                               [0, 0, 0, 0, 0.00005, 0],
                               [0, 0, 0, 0, 0, 0.0001]])
            gRp = landmark_tf.rotation().matrix()
            P = my_cov[3:6, 3:6]
            transformed_cov = gRp.T @ P @ gRp
            my_cov[3:6, 3:6] = transformed_cov
            noise: gtsam.noiseModel.Gaussian = gtsam.noiseModel.Gaussian.Covariance(my_cov)
            # cov = noise.covariance()

            graph.add(gtsam.BetweenFactorPose3(previous_key, landmark_key, noisy_landmark_tf, noise))
            # tf = gtsam.Unit3(np.array((noisy_landmark_tf.x(), noisy_landmark_tf.y(), noisy_landmark_tf.z())))
            # graph.add(gtsam.BearingRangeFactor3D(previous_key, landmark_key, noisy_landmark_tf, 2, ODOMETRY_NOISE))
            # gt_dict[str(Symbol(landmark_key).string())] = true_landmarks[l]
            if landmark_key not in detected_landmarks:
                detected_landmarks.add(landmark_key)
                noisy_estimate = current_estimate.atPose3(previous_key).compose(noisy_landmark_tf)
                initial_estimate.insert(landmark_key, noisy_estimate)

        # Compute and insert the initialization estimate for the current pose using a noisy odometry measurement.
        # Perform incremental update to iSAM2's internal Bayes tree, optimizing only the affected variables.
        isam.update(graph, initial_estimate)
        current_estimate = isam.calculateEstimate()
        estimate_frames.append(estimate_to_dict(graph, current_estimate, landmark_keys, current_key))
        # Report all current state estimates from the iSAM2 optimization.
        # print(gt_dict)
        draw_3d_estimate(graph, current_estimate, False)

        initial_estimate.clear()

    # Print the final covariance matrix for each pose after completing inference.
    marginals = gtsam.Marginals(graph, current_estimate)
    i = 1
    while current_estimate.exists(i):
        print(f"X{i} covariance:\n{marginals.marginalCovariance(i)}\n")
        i += 1
    save_data(dataset_path/"frames_refined_prediction.p", estimate_frames)

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    Pose3_ISAM2_example()