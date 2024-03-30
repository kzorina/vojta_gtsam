import numpy as np
import gtsam
from typing import Optional, List
import pinocchio as pin


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


def plus_se3_local(T, nu, dt):
    return T * gtsam.Pose3.Expmap(nu * dt)


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


def plus_se3_global(T, nu, dt):
    return gtsam.Pose3.Expmap(nu * dt) * T


def error_velocity_integration_so3r3_global(
        dt: float, this: gtsam.CustomFactor,
        values: gtsam.Values,
        jacobians: Optional[List[np.ndarray]]) -> np.ndarray:
    """Odometry Factor error function

    Encodes relationship: 
    nu12 = [o12, v12] 
    R2 = Exp3(o12*dt) * R1 
    p2 = p1 + v12*dt

    as the error:
    eo = Log(R2 * R1.T) - o12*dt
    et = p2 - p1 - nu12*dt
    e = [eo, et]

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

    R1, R2 = T1.rotation(), T2.rotation()
    p1, p2 = T1.translation(), T2.translation()
    o12, v12 = nu12[:3], nu12[3:]

    # Compute error
    R12 = R2 * R1.inverse()
    # error_o = gtsam.Rot3.Logmap(R12) - o12*dt
    error_o = pin.log3(R12.matrix()) - o12 * dt
    error_p = p2 - p1 - v12 * dt
    # and jacobians
    if jacobians is not None:
        # TODO: could not find gtsam equivalent
        JlogR12 = pin.Jlog3(R12.matrix())

        jacobians[0] = np.zeros((6, 6))
        jacobians[0][:3, :3] = JlogR12 @ (-R1.matrix())
        jacobians[0][3:6, 3:6] = -R1.matrix()
        jacobians[1] = np.zeros((6, 6))
        jacobians[1][:3, :3] = JlogR12 @ R1.matrix()
        jacobians[1][3:6, 3:6] = R2.matrix()
        jacobians[2] = -np.eye(6) * dt

    return np.concatenate([error_o, error_p])


def plus_so3r3_global(T: gtsam.Pose3, nu: np.ndarray, dt: float):
    o, v = nu[:3], nu[3:]
    R_new = gtsam.Rot3.Expmap(o * dt) * T.rotation()
    t_new = T.translation() + v * dt
    # return gtsam.Pose3(gtsam.Rot3.Expmap(do*dt), dv*dt) * T
    return gtsam.Pose3(R_new, t_new)