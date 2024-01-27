import numpy as np
import yaml


def compute_covariance(t, R, K, a_points, sig_pix):
    J_cproj_TR = points_projection_jac(t, R, K, a_points)
    cov_tR = sig_pix**2 * np.linalg.inv(J_cproj_TR.T @ J_cproj_TR)
    return cov_tR


def points_projection(t, R, K, a_points):
    n = a_points.shape[0]
    # stack of points projections
    X = np.zeros((2*n)) 
    
    for i in range(n):
        p = a_points[i,:]
        # Pinhole projection to non normalized homogeneous coordinates in pixels
        h = pinhole(t, R, K,  p)
        # Euclidianization
        u = h2pix(h)
        X[2*i:2*i+2] = u

    return X

def points_projection_jac(t, R, K, a_points):
    n = a_points.shape[0]
    # jacobian of the corner projections wrt. relative transformation
    J_cproj_TR = np.zeros((2*n,6))

    for i in range(n):
        p = a_points[i,:]
        # Pinhole projection to non normalized homogeneous coordinates in pixels (along with jacobians)
        h = pinhole(t, R, K,  p)
        J_h_TR = pinhole_jac(t, R, K,  p)
        # Euclidianization Jacobian
        J_u_h = h2pix_jac(h)
        # Fill tag projection jacobian for ith corner
        J_cproj_TR[2*i:2*i+2,:] = J_u_h @ J_h_TR

    return J_cproj_TR

def se3_points_projection_jac(t, R, K, a_points):
    n = a_points.shape[0]
    # jacobian of the corner projections wrt. relative transformation
    J_cproj_TR = np.zeros((2*n, 6))

    for i in range(n):
        p = a_points[i, :]
        # Pinhole projection to non normalized homogeneous coordinates in pixels (along with jacobians)
        h = pinhole(t, R, K, p)
        J_h_TR = se3_pinhole_jac(t, R, K, p)
        # Euclidianization Jacobian
        J_u_h = h2pix_jac(h)
        # Fill tag projection jacobian for ith corner
        J_cproj_TR[2*i:2*i+2,:] = J_u_h @ J_h_TR

    return J_cproj_TR

def h2pix(h):
    """
    Transform the homogeneous representation of projected point to 2D pixel and compute jacs.    
    """
    return np.array([h[0]/h[2], h[1]/h[2]])

def h2pix_jac(h):
    """
    Transform the homogeneous representation of projected point to 2D pixel and compute jacs.    
    """

    J_u_h = np.array([
        [1 / h[2], 0, -h[0]/(h[2]*h[2])],
        [0, 1.0/h[2], -h[1]/(h[2]*h[2])],
    ])
    return J_u_h

def pinhole(t, R, K,  p):
    """
    K: camera matrix
    t = c_p_ca: translation from camera to 3d point in camera frame
    R = c_R_a: rotation from camera to 3d point in camera frame
    p = a_p: 3D point in object frame
    """
    return K @ (t + R @ p)


def pinhole_jac(t, R, K, p):
    J_h_T = K
    p_hat = np.array([
        [ 0, -p[2], p[1],],
        [ p[2], 0, -p[0],],
        [-p[1], p[0], 0  ],
    ]) 
    J_h_R = -K @ R @ p_hat

    # 3 x 6 tag to camera translation|rotation jacobian
    return np.hstack([J_h_T, J_h_R])
        
def se3_pinhole_jac(t, R, K, p):
    J_h_T = K@R
    p_hat = np.array([
        [ 0, -p[2], p[1],],
        [ p[2], 0, -p[0],],
        [-p[1], p[0], 0  ],
    ])
    J_h_R = -K @ R @ p_hat

    # 3 x 6 tag to camera translation|rotation jacobian
    return np.hstack([J_h_T, J_h_R])

def get_cam_model(path):
    """
    Assuming ros calibration convention
    """
    with open(path) as f:
        parsed = yaml.load(f, Loader=yaml.FullLoader)

    width = parsed['width']
    height = parsed['height']
    mat = parsed['projection_matrix']

    K = np.array(mat['data']).reshape((mat['rows'], mat['cols']))[:,:3]

    return width, height, K
    
def get_cam_model_cv(path):
    with open(path) as f:
        parsed = yaml.load(f, Loader=yaml.FullLoader)

    width = parsed['image_width']
    height = parsed['image_height']
    mat = parsed['camera_matrix']

    K = np.array(mat['data']).reshape((mat['rows'], mat['cols']))

    return width, height, K


if __name__ == '__main__':
    import pinocchio as pin


    tag_width = 0.1

    # Order of the corners: anti clockwise, looking at the tag, starting from bottom left.
    # Looking at the tag, the reference frame is
    # X = Right, Y = Down, Z = Inside the plane -> identity transformation when looking straight/fronto parallell at it
    # corner coordinates in the tag frame
    # https://github.com/AprilRobotics/apriltag/blob/04c4ec3fbb9cbbec1344049323d24e752f01bf38/apriltag_detect.docstring
    a_corners = 0.5*tag_width*np.array([
        [-1,  1, 0], # bottom left
        [ 1,  1, 0], # bottom right
        [ 1, -1, 0], # top right
        [-1, -1, 0], # top left
    ])

    # camera matrix
    # K = [fx, 0, cx
    #      0, fy, cy
    #      0, 0,  1]
    K = np.array([
        [30,0, 5],
        [0, 20,4],
        [0, 0, 1],
    ])

    # T = pin.SE3.Identity()
    T = pin.SE3.Random()
    t, R = T.translation, T.rotation
    # t[0] = 1
    # t[1] = 1
    t[2] = 2
    R = pin.rpy.rpyToMatrix(0.1, -0.0, 0.1)

    # J = points_projection_jac(t, R, K, a_corners)
    sig_pix = 1
    Q = compute_covariance(t, R, K, a_corners, sig_pix)
    print('Q')
    print(Q)
