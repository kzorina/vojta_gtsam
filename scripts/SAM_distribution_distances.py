import numpy as np


def mahalanobis_distance(x, covariance):
    assert (x.shape[0] == covariance.shape[0])
    assert (covariance.shape[0] == covariance.shape[1])
    cov_inv = np.linalg.inv(covariance)
    v = x
    return np.sqrt(v@cov_inv@v)

def bhattacharyya_distance(x, covariance_t, covariance_x):
    assert (x.shape[0] == covariance_t.shape[0] == covariance_x.shape[0])
    assert (covariance_t.shape[0] == covariance_t.shape[1])
    assert (covariance_x.shape[0] == covariance_x.shape[1])
    cov = (covariance_t + covariance_x)/2
    cov_inv = np.linalg.inv(cov)
    cov_det = np.linalg.det(cov)
    cov_t_det = np.linalg.det(covariance_t)
    cov_x_det = np.linalg.det(covariance_x)
    v = x
    return (v@cov_inv@v)/8 + np.log(cov_det/(np.sqrt(cov_x_det*cov_t_det)))/2