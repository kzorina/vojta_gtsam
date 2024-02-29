from typing import Optional, List
import numpy as np
import gtsam


def error_velocity_integration_local(
        dt: float, this: gtsam.CustomFactor,
        values: gtsam.Values,
        jacobians: Optional[List[np.ndarray]]) -> np.ndarray:
    """Odometry Factor error function

    Encodes relationship: T2 = T1 * Exp(nu12*dt)
    as the error:
    e = Log(T1.inv() * T2) - nu12*dt

    :param measurement: [dt]!
    :param this: gtsam.CustomFactor handle
    :param values: gtsam.Values
    :param jacobians: Optional list of Jacobians
    :return: the unwhitened error
    """

    # Retrieve values
    key_p1 = this.keys()[0]
    key_p2 = this.keys()[1]
    key_v1 = this.keys()[2]
    T1, T2 = values.atPose3(key_p1), values.atPose3(key_p2)
    nu12 = values.atVector(key_v1)

    # Compute error
    M12 = T1.inverse() * T2
    error = gtsam.Pose3.Logmap(M12) - nu12 * dt
    # and jacobians
    if jacobians is not None:
        JlogM12 = np.eye(6, order='F')  # Logmap only accepts f_continuous Jac array
        gtsam.Pose3.Logmap(M12, JlogM12)

        jacobians[0] = JlogM12 @ T2.inverse().AdjointMap() @ (-T1.AdjointMap())
        jacobians[1] = JlogM12
        jacobians[2] = -np.eye(6) * dt

    return error


def error_velocity_integration_global(
        dt: float, this: gtsam.CustomFactor,
        values: gtsam.Values,
        jacobians: Optional[List[np.ndarray]]) -> np.ndarray:
    """Odometry Factor error function

    Encodes relationship: T2 = Exp(nu12*dt) * T1
    as the error:
    e = Log(T2 * T1.inv()) - nu12*dt

    :param measurement: [dt]!
    :param this: gtsam.CustomFactor handle
    :param values: gtsam.Values
    :param jacobians: Optional list of Jacobians
    :return: the unwhitened error
    """

    # Retrieve values
    key_p1 = this.keys()[0]
    key_p2 = this.keys()[1]
    key_v1 = this.keys()[2]
    T1, T2 = values.atPose3(key_p1), values.atPose3(key_p2)
    nu12 = values.atVector(key_v1)

    # Compute error
    M12 = T2 * T1.inverse()
    error = gtsam.Pose3.Logmap(M12) - nu12 * dt
    # and jacobians
    if jacobians is not None:
        JlogM12 = np.eye(6, order='F')  # Logmap only accepts f_continuous Jac array
        gtsam.Pose3.Logmap(M12, JlogM12)
        jacobians[0] = JlogM12 @ (-T1.AdjointMap())
        jacobians[1] = JlogM12 @ T1.AdjointMap()
        jacobians[2] = -np.eye(6) * dt

    return error