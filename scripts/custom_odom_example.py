
import numpy as np
import gtsam
from typing import List
from functools import partial
import graphviz

from custom_odom_factors import error_velocity_integration_local, error_velocity_integration_global

"""
Simulate series of object with absolute pose measurements and cst velocity model.
State = [Two, nu]
nu: se(3) twist either in local coordinates or global coordinates

3 types of factors:
- Absolute pose: gtsam.PriorFactorPose3
- Relative velocity: ?
- Velocity integration: Custom
"""

def simulate_rigid_body_motion(dt, N_traj, local=True) -> List[gtsam.Pose3]:
    """
    Simulate a rigid body motion
    output:
    - p_lst: size N
    - nu_lst: size N-1
    s.t. p[i+1] = p[i]*Exp(nu[i])
    """
    p0 = gtsam.Pose3.Identity()
    p_lst = [p0]
    nu_lst = []
    p = p0
    for _ in range(N_traj-1):
        nu = 0.1*np.ones(6)
        if local:
            p = p * gtsam.Pose3.Expmap(nu*dt)
        else:
            p = gtsam.Pose3.Expmap(nu*dt) * p
        p_lst.append(p)
        nu_lst.append(nu)
    nu_lst.append(nu.copy())
    return p_lst, nu_lst


if __name__ == '__main__':
    DT = 0.25
    N_traj = 10
    p_lst, nu_lst = simulate_rigid_body_motion(DT, N_traj)

    LOCAL_TWIST = False
    noisy_meas = False  # set this to False to run with "perfect" measurements
    noisy_init = True  # set this to False to run with "perfect" initialization

    # absolute pose measurements
    sigma_se3_m = 0.01*np.ones(6)  # assume GPS is +/- 3m
    sigma_se3_init = 1*np.ones(6)  # assume GPS is +/- 3m
    poses_m = [
        p_lst[k] * gtsam.Pose3.Expmap(np.random.normal(scale=sigma_se3_m) if noisy_meas else np.zeros(6))
        for k in range(N_traj)
    ]
    pose_init = [
        p_lst[k] * gtsam.Pose3.Expmap(np.random.normal(scale=sigma_se3_init) if noisy_init else np.zeros(6))
        for k in range(N_traj)
    ]
    twist_init = [
        nu_lst[k] + (np.random.normal(scale=sigma_se3_init) if noisy_init else np.zeros(6))
        for k in range(N_traj)
    ]

    pose_unknown = [gtsam.symbol('p', k) for k in range(N_traj)]
    twist_unknown = [gtsam.symbol('v', k) for k in range(N_traj)]

    print("pose_unknown = ", list(map(gtsam.DefaultKeyFormatter, pose_unknown)))
    print("twist_unknown = ", list(map(gtsam.DefaultKeyFormatter, twist_unknown)))

    # We now can use nonlinear factor graphs
    factor_graph = gtsam.NonlinearFactorGraph()

    # Add absolute pose factors
    prior_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.01)
    # for i in range(N_traj):
    for i in range(2):
        factor_graph.push_back(gtsam.PriorFactorPose3(pose_unknown[i], poses_m[i], prior_noise))

    # Add constant twist factors
    prior_cst_twist = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
    for i in range(N_traj-1):
        factor_graph.push_back(gtsam.BetweenFactorVector(twist_unknown[i], twist_unknown[i+1], np.zeros(6), prior_cst_twist))

    # Add twist integration factor
    prior_int_twist = gtsam.noiseModel.Isotropic.Sigma(6, 1000.00)
    for i in range(N_traj-1):
        if LOCAL_TWIST:
            error_func = partial(error_velocity_integration_local, DT)
        else:
            error_func = partial(error_velocity_integration_global, DT)
        fint = gtsam.CustomFactor(
            prior_int_twist,
            [pose_unknown[i],pose_unknown[i+1],twist_unknown[i]],
            error_func
            )
        factor_graph.push_back(fint)

    # New Values container
    v_init = gtsam.Values()

    # Add initial estimates to the Values container
    for i in range(N_traj):
        v_init.insert(pose_unknown[i], pose_init[i])
        v_init.insert(twist_unknown[i], twist_init[i])

    # Initialize optimizer
    params = gtsam.GaussNewtonParams()
    optimizer = gtsam.GaussNewtonOptimizer(factor_graph, v_init, params)

    # Optimize the factor graph
    print("\n\n INIT")
    print(v_init)
    v_final = optimizer.optimize()
    print("\n\n FINAL")
    print(v_final)
    marginals = gtsam.Marginals(factor_graph,v_final)
    print("hola:", marginals.marginalCovariance(pose_unknown[0]))
    # # calculate the error from ground truth
    # error_p = np.array([gtsam.Pose3.Logmap(result.atPose3(pose_unknown[k]).inverse() * p_lst[k])
    #                   for k in range(5)])
    # print(error_p)
    error_v = np.array([v_final.atVector(twist_unknown[k]) - nu_lst[k]
                      for k in range(5)])
    print("\n\n error_v")
    print(error_v)