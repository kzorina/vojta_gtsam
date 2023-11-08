from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LinearLocator

import gtsam
import scipy
import gtsam.utils.plot as gtsam_plot
from Vizualize import draw_3d_estimate


def probability_density(x, centre, covariance, covariance_x):
    assert(x.shape == centre.shape)
    assert (x.shape[0] == centre.shape[0] == covariance.shape[0])
    assert (covariance.shape[0] == covariance.shape[1])
    cov_inv = np.linalg.inv(covariance)
    cov_det = np.linalg.det(covariance)
    k = x.shape[0]
    ret = np.exp(-0.5*(x - centre).T@cov_inv@(x - centre))/np.sqrt(cov_det*(2*np.pi)**k)
    return ret

def mahalanobis_distance(x, centre, covariance, covariance_x):
    assert(x.shape == centre.shape)
    assert (x.shape[0] == centre.shape[0] == covariance.shape[0])
    assert (covariance.shape[0] == covariance.shape[1])
    cov_inv = np.linalg.inv(covariance)
    v = centre - x
    return np.sqrt(v@cov_inv@v)

def bhattacharyya_distance(x, centre, covariance_t, covariance_x):
    assert(x.shape == centre.shape)
    assert (x.shape[0] == centre.shape[0] == covariance_t.shape[0] == covariance_x.shape[0])
    assert (covariance_t.shape[0] == covariance_t.shape[1])
    assert (covariance_x.shape[0] == covariance_x.shape[1])
    cov = (covariance_t + covariance_x)/2
    cov_inv = np.linalg.inv(cov)
    cov_det = np.linalg.det(cov)
    cov_t_det = np.linalg.det(covariance_t)
    cov_x_det = np.linalg.det(covariance_x)
    v = x - centre
    return (v@cov_inv@v)/8 + np.log(cov_det/(np.sqrt(cov_x_det*cov_t_det)))/2

def kullback_leibler_divergence(x, centre, covariance_t, covariance_x):
    assert(x.shape == centre.shape)
    assert (x.shape[0] == centre.shape[0] == covariance_t.shape[0] == covariance_x.shape[0])
    assert (covariance_t.shape[0] == covariance_t.shape[1])
    assert (covariance_x.shape[0] == covariance_x.shape[1])
    cov = (covariance_t + covariance_x)/2
    cov_inv = np.linalg.inv(cov)
    cov_x_inv = np.linalg.inv(covariance_x)
    cov_det = np.linalg.det(cov)
    cov_t_det = np.linalg.det(covariance_t)
    cov_x_det = np.linalg.det(covariance_x)
    v = x - centre
    k = x.shape[0]
    return ((v@cov_inv@v) - k + np.trace(cov_x_inv@covariance_t) + np.log(cov_x_det/cov_t_det))/2

def get_distribution_grid(function, centre, covariance, covariance_x, min_xy = -3, max_xy = 3, step_size = 0.1):
    X = np.arange(min_xy, max_xy, step_size)
    Y = np.arange(min_xy, max_xy, step_size)
    P = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            p = function(np.array([X[i], Y[j]]), centre, covariance, covariance_x)
            P[i, j] = p
    Y, X = np.meshgrid(X, Y)
    return X, Y, P

if __name__ == '__main__':
    ax = plt.figure().add_subplot(projection='3d')
    centre = np.array([0, 0])
    covariance = np.array([[3, 1],
                           [1, 1]])
    covariance_x = np.array([[6, 2],
                           [2, 4]])
    min_xy = -3
    max_xy = 3
    step_size = 0.1
    # X, Y, P = get_distribution_grid(probability_density, centre, covariance, covariance_x, min_xy = min_xy, max_xy = max_xy, step_size = step_size)
    # X, Y, P = get_distribution_grid(mahalanobis_distance, centre, covariance, covariance_x, min_xy = min_xy, max_xy = max_xy, step_size = step_size)
    X, Y, P = get_distribution_grid(bhattacharyya_distance, centre, covariance, covariance_x, min_xy = min_xy, max_xy = max_xy, step_size = step_size)
    # X, Y, P = get_distribution_grid(kullback_leibler_divergence, centre, covariance, covariance_x, min_xy = min_xy, max_xy = max_xy, step_size = step_size)

    colors = np.empty(X.shape, dtype=str)
    surf = ax.plot_surface(X, Y, P, alpha=0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_zlim(-4, 4)

    plt.show()


[]