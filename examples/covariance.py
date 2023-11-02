
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LinearLocator

import gtsam
import scipy
import gtsam.utils.plot as gtsam_plot
from Vizualize import draw_3d_estimate


def probability_density(x, centre, covariance):
    assert(x.shape == centre.shape)
    assert (x.shape[0] == centre.shape[0] == covariance.shape[0])
    assert (covariance.shape[0] == covariance.shape[1])
    cov_inv = np.linalg.inv(covariance)
    cov_det = np.linalg.det(covariance)
    k = x.shape[0]
    ret = np.exp(-0.5*(x - centre).T@cov_inv@(x - centre))/np.sqrt(cov_det*(2*np.pi)**k)
    return ret

if __name__ == '__main__':
    # my_cov = np.array([[0.001, 0, 0, 0, 0, 0],
    #                    [0, 0.001, 0, 0, 0, 0],
    #                    [0, 0, 0.001, 0, 0, 0],
    #                    [0, 0, 0, 0.005, 0, 0],
    #                    [0, 0, 0, 0, 0.005, 0],
    #                    [0, 0, 0, 0, 0, 0.01]])
    # x_points = np.arange(-3, 3, 0.1)
    # p_points = np.zeros(x_points.shape[0])
    # centre = np.array([0])
    # covariance = np.array([[1]])
    # for i in range(x_points.shape[0]):
    #     p = probability_density(np.array([x_points[i]]), centre, covariance)
    #     p_points[i] = p
    # plt.plot(x_points, p_points)
    # plt.show()
    ax = plt.figure().add_subplot(projection='3d')

    X = np.arange(-3, 3, 0.1)
    Y = np.arange(-3, 3, 0.1)
    P = np.zeros((X.shape[0], X.shape[0]))
    centre = np.array([0, 0])
    covariance = np.array([[0.001, 0],
                           [0, 0.001]])
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            p = probability_density(np.array([X[i], Y[j]]), centre, covariance)
            P[i, j] = p

    X, Y = np.meshgrid(X, Y)
    colortuple = ('b', 'g')
    colors = np.empty(X.shape, dtype=str)
    for y in range(Y.shape[0]):
        for x in range(X.shape[0]):
            colors[y, x] = colortuple[(x + y) % len(colortuple)]
    print(probability_density(np.array([0, 0.1]), centre, covariance))
    surf = ax.plot_surface(X, Y, P, alpha=0.8)
    # ax.set_zlim(-1, 1)
    # ax.zaxis.set_major_locator(LinearLocator(6))
    # plt.plot(x_points, p_points)
    plt.show()


[]