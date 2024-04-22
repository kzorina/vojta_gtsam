import numpy as np
import unittest
import time
np.random.seed(0)


def random_cov(dim=3, sigma=0.2):
    A = np.random.rand(dim, dim)
    Q = sigma**2*np.dot(A, A.T)
    return Q

def frobenius_norm(Q1, Q2):
    # Distance measure between two matrices
    return np.sqrt(np.trace((Q1 - Q2).T @ (Q1 - Q2)))

class TestExtrapolation(unittest.TestCase):

    def test_velocity_integration(self):
        N_samples = int(1e4)
        D = 3
        for a in range(1):
            Q = random_cov(dim=D, sigma=1)

            time_duration = 2  # s
            samples = np.zeros((N_samples, D))
            Q_est = Q * time_duration**2
            for x in range(N_samples):
                v = np.random.multivariate_normal(np.zeros(D), Q)
                t = v * time_duration
                samples[x, :] = t
            #     print(f"({x}/{N_samples})", end='\r')
            # print("")
            Q_num = np.cov(samples, rowvar=False)
            assert frobenius_norm(Q_est, Q_num) < 10**(0)

    def test_acc_integration(self):
        N_samples = int(1e4)
        D = 3
        for a in range(1):
            Q = random_cov(dim=D, sigma=1)

            time_duration = 2  # s
            samples = np.zeros((N_samples, D))
            Q_est = (Q * time_duration**4)/4
            for x in range(N_samples):
                a = np.random.multivariate_normal(np.zeros(D), Q)
                t = (a * time_duration**2)/2
                samples[x, :] = t
            #     print(f"({x}/{N_samples})", end='\r')
            # print("")
            Q_num = np.cov(samples, rowvar=False)
            assert frobenius_norm(Q_est, Q_num) < 10**(0)

    def test_jerk_integration(self):
        N_samples = int(1e4)
        D = 3
        for a in range(1):
            Q = random_cov(dim=D, sigma=1)

            time_duration = 2  # s
            samples = np.zeros((N_samples, D))
            Q_est = (Q * time_duration**6)/36
            for x in range(N_samples):
                j = np.random.multivariate_normal(np.zeros(D), Q)
                t = (j * time_duration**3)/6
                samples[x, :] = t
            #     print(f"({x}/{N_samples})", end='\r')
            # print("")
            Q_num = np.cov(samples, rowvar=False)
            assert frobenius_norm(Q_est, Q_num) < 10**(0)

    def test_brownian_motion_jerk_wiener(self):
        N_samples = int(1e4)
        D = 1
        start_time = time.time()
        for a in range(1):
            Q = random_cov(dim=D, sigma=1)
            time_duration = 200  # s
            n = 100
            samples = np.zeros((N_samples, D))
            Q_est = (Q * time_duration ** 7)/250
            etas = np.random.multivariate_normal(np.zeros(D), Q, size=(N_samples, time_duration * n))
            for x in range(N_samples):
                t = np.zeros((D))
                v = np.zeros((D))
                a = np.zeros((D))
                j = np.zeros((D))
                t2 = np.sum(np.cumsum(np.cumsum(np.cumsum(etas[x, :, :])/np.sqrt(n))/np.sqrt(n))/np.sqrt(n))
                # t3 = (np.sum(etas[x, :, :]) * time_duration ** 1)/2
                # for i in range(time_duration):
                #     # eta = np.random.multivariate_normal(np.zeros(D), Q)
                #     eta = etas[x, i, :]
                #     j += eta
                #     a += j
                #     v += a
                #     t += v
                samples[x, :] = t2
            #     print(f"({x}/{N_samples})", end='\r')
            # print("")
            Q_num = np.cov(samples, rowvar=False)[np.newaxis]
            elt = time.time() - start_time
            assert frobenius_norm(Q_est, Q_num) < 4

    def test_brownian_motion_jerk(self):
        N_samples = int(1e4)
        D = 1
        start_time = time.time()
        for a in range(1):
            Q = random_cov(dim=D, sigma=1)
            time_duration = 200  # s
            samples = np.zeros((N_samples, D))
            Q_est = (Q * time_duration ** 7)/250
            etas = np.random.multivariate_normal(np.zeros(D), Q, size=(N_samples, time_duration))
            for x in range(N_samples):
                t = np.zeros((D))
                v = np.zeros((D))
                a = np.zeros((D))
                j = np.zeros((D))
                t2 = np.sum(np.cumsum(np.cumsum(np.cumsum(etas[x, :, :]))))
                # t3 = (np.sum(etas[x, :, :]) * time_duration ** 1)/2
                # for i in range(time_duration):
                #     # eta = np.random.multivariate_normal(np.zeros(D), Q)
                #     eta = etas[x, i, :]
                #     j += eta
                #     a += j
                #     v += a
                #     t += v
                samples[x, :] = t2
            #     print(f"({x}/{N_samples})", end='\r')
            # print("")
            Q_num = np.cov(samples, rowvar=False)[np.newaxis]
            elt = time.time() - start_time
            assert frobenius_norm(Q_est, Q_num) < 4

    def test_brownian_motion_acc(self):
        N_samples = int(1e5)
        D = 1
        start_time = time.time()
        for a in range(1):
            Q = random_cov(dim=D, sigma=1)
            time_duration = 900  # s
            samples = np.zeros((N_samples, D))
            Q_est = (Q * time_duration ** 5)/20
            etas = np.random.multivariate_normal(np.zeros(D), Q, size=(N_samples, time_duration))
            for x in range(N_samples):
                t = np.zeros((D))
                v = np.zeros((D))
                a = np.zeros((D))
                t2 = np.sum(np.cumsum(np.cumsum(etas[x, :, :])))
                # t3 = (np.sum(etas[x, :, :]) * time_duration ** 1)/2
                # for i in range(time_duration):
                #     # eta = np.random.multivariate_normal(np.zeros(D), Q)
                #     eta = etas[x, i, :]
                #     a += eta
                #     v += a
                #     t += v
                samples[x, :] = t2
            #     print(f"({x}/{N_samples})", end='\r')
            # print("")
            Q_num = np.cov(samples, rowvar=False)[np.newaxis]
            elt = time.time() - start_time
            assert frobenius_norm(Q_est, Q_num) < 4

    def test_brownian_motion_vel(self):
        N_samples = int(1e4)
        D = 1
        start_time = time.time()
        for a in range(1):
            Q = random_cov(dim=D, sigma=1)
            time_duration = 900  # s
            samples = np.zeros((N_samples, D))
            Q_est = (Q * time_duration ** 3)/3
            etas = np.random.multivariate_normal(np.zeros(D), Q, size=(N_samples, time_duration))
            for x in range(N_samples):
                t = np.zeros((D))
                v = np.zeros((D))
                t2 = np.sum(np.cumsum(etas[x, :, :]))
                t3 = (np.sum(etas[x, :, :]) * time_duration ** 1)/2
                # for i in range(time_duration):
                #     # eta = np.random.multivariate_normal(np.zeros(D), Q)
                #     eta = etas[x, i, :]
                #     v += eta
                #     t += v
                samples[x, :] = t3
            #     print(f"({x}/{N_samples})", end='\r')
            # print("")
            Q_num = np.cov(samples, rowvar=False)[np.newaxis]
            elt = time.time() - start_time
            assert frobenius_norm(Q_est, Q_num) < 4

    def test_brownian_motion(self):
        N_samples = int(1e3)
        D = 1
        for a in range(1):
            Q = random_cov(dim=D, sigma=1)
            time_duration = 100  # s
            samples = np.zeros((N_samples, D))
            Q_est = Q * time_duration
            for x in range(N_samples):
                t = np.zeros((D))
                for i in range(time_duration):
                    # eta = np.random.multivariate_normal(np.zeros(D), Q)
                    eta = np.random.multivariate_normal(np.zeros(D), Q)
                    t += eta
                samples[x, :] = t
            #     print(f"({x}/{N_samples})", end='\r')
            # print("")
            Q_num = np.cov(samples, rowvar=False)
            assert frobenius_norm(Q_est, Q_num) < 4


    def test_wiener_process(self):
        N_samples = int(1e3)
        D = 1
        for a in range(1):
            Q = random_cov(dim=D, sigma=1)

            time_duration = 3  # s
            steps_count = 100
            samples = np.zeros((N_samples, D))
            Q_est = Q * time_duration
            for x in range(N_samples):
                t = np.zeros((D))
                for i in range(time_duration * steps_count):
                    # eta = np.random.multivariate_normal(np.zeros(D), Q)
                    dt = 1 / np.sqrt(steps_count)
                    eta = np.random.multivariate_normal(np.zeros(D), Q)
                    t += eta * dt
                samples[x, :] = t
            #     print(f"({x}/{N_samples})", end='\r')
            # print("")
            Q_num = np.cov(samples, rowvar=False)
            assert frobenius_norm(Q_est, Q_num) < 10**(-2)



if __name__ == "__main__":
    unittest.main()
    print("Everything passed")