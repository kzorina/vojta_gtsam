import gtsam
import numpy as np
from functools import partial
from custom_odom_factors import *
import matplotlib.pyplot as plt
import matplotlib

def simulate_cov_growth(sigma=1, dt=(1/30), T=1.0, triple_sigma=10**(-7), vel_prior_sigma=0.1):
    parameters = gtsam.ISAM2Params()
    parameters.setRelinearizeThreshold(0.1)
    initial_estimate = gtsam.Values()
    new_graph = gtsam.NonlinearFactorGraph()
    isam = gtsam.ISAM2(parameters)
    SYMBOL_GAP = 10**6

    # noise = gtsam.noiseModel.Isotropic.Sigma(6, sigma)
    triple_factor_noise = gtsam.noiseModel.Isotropic.Sigma(6, triple_sigma)

    new_graph.add(gtsam.PriorFactorVector(0, np.zeros((6)), triple_factor_noise))
    # new_graph.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(np.eye(4)), triple_factor_noise))

    initial_estimate.insert(0, np.zeros((6)))
    # initial_estimate.insert(0, gtsam.Pose3(np.eye(4)))
    # T = 1
    N = int(T/(dt))
    # N = 30
    # dt = 1/30

    error_func = partial(error_derivative_integration_so3r3_global, dt)
    # error_func = partial(error_derivative_integration_so3r3_global_new, dt)
    # error_func = partial(error_velocity_integration_so3r3_global, dt)

    samplesQ = np.full((N), None)
    samplesQv = np.full((N), None)
    Qs = np.zeros((N, 6, 6))
    Qvs = np.zeros((N, 6, 6))
    dts = np.zeros((N))
    for i in range(1, N):
        dts[i] = i*dt
        new_graph.add(gtsam.CustomFactor(triple_factor_noise, [i-1, i, SYMBOL_GAP + i], error_func))
        initial_estimate.insert(SYMBOL_GAP + i, np.zeros((6)))
        if i == 1:
            # new_graph.add(gtsam.PriorFactorVector(SYMBOL_GAP + i, np.zeros((6)), triple_factor_noise))
            new_graph.add(gtsam.PriorFactorVector(SYMBOL_GAP + i, np.zeros((6)), gtsam.noiseModel.Gaussian.Covariance(np.eye(6) * vel_prior_sigma)))
        if i > 1:
            new_graph.add(gtsam.BetweenFactorVector(SYMBOL_GAP + i - 1, SYMBOL_GAP + i, np.zeros(6), gtsam.noiseModel.Gaussian.Covariance(sigma*np.eye(6) * dt)))
        initial_estimate.insert(i, np.zeros((6)))
        # initial_estimate.insert(i, gtsam.Pose3(np.eye(4)))

        isam.update(new_graph, initial_estimate)
        current_estimate = isam.calculateEstimate()
        Q = isam.marginalCovariance(i)
        Qv = isam.marginalCovariance(SYMBOL_GAP + i)
        new_graph = gtsam.NonlinearFactorGraph()
        initial_estimate.clear()
        # samplesQ[i] = np.linalg.det(Q[:3, :3]) ** (1/3)
        # samplesQv[i] = np.linalg.det(Qv[:3, :3]) ** (1/3)
        samplesQ[i] = np.linalg.det(Q) ** (1/6)
        samplesQv[i] = np.linalg.det(Qv) ** (1/6)
        Qs[i, :, :] = Q
        Qvs[i, :, :] = Qv
        if i*dt > 3:
            pass
    return dts, samplesQ, samplesQv

def simulate_velocity_estimate(error_func, sigma, dt=(1/30)):
    parameters = gtsam.ISAM2Params()
    parameters.setRelinearizeThreshold(0.1)
    initial_estimate = gtsam.Values()
    new_graph = gtsam.NonlinearFactorGraph()
    isam = gtsam.ISAM2(parameters)
    SYMBOL_GAP = 10**4
    velocity = np.array((1, 0, 0, 0, 0, 0))

    noise = gtsam.noiseModel.Gaussian.Covariance(np.eye(6) * sigma**2)
    triple_factor_noise = gtsam.noiseModel.Isotropic.Sigma(6, 10 ** (-7))

    new_graph.add(gtsam.PriorFactorVector(0, velocity * 0, noise))
    initial_estimate.insert(0, velocity * 0)

    new_graph.add(gtsam.PriorFactorVector(1, velocity * dt, noise))
    initial_estimate.insert(1, velocity * dt)

    new_graph.add(gtsam.CustomFactor(triple_factor_noise, [0, 1, SYMBOL_GAP + 0], error_func))
    initial_estimate.insert(SYMBOL_GAP, np.zeros((6)))

    isam.update(new_graph, initial_estimate)
    current_estimate = isam.calculateEstimate()
    Qv = isam.marginalCovariance(SYMBOL_GAP)
    Q0 = isam.marginalCovariance(0)
    Q1 = isam.marginalCovariance(1)
    return Qv

def simulate_velocity_cov(error_func, sigma, N=100):
    arr = np.zeros((N))
    dts = np.zeros((N))
    for i in range(0, N):
        dt = 0.1 + (i) / N
        dts[i] = dt
        err_func = partial(error_func, dt)
        arr[i] = np.linalg.det(simulate_velocity_estimate(err_func, sigma, dt))**(1/6)
    return dts, arr

def simulate_num_vel_estimate(sigma):
    D = 3
    Q = np.eye((D)) * sigma ** 2
    N = 1000
    dts_count = 200
    vel = np.random.rand((D))
    dts = np.zeros((dts_count))
    results = np.zeros((dts_count))
    for i in range(dts_count):
        dt = 0.1 + i/dts_count
        dts[i] = dt
        etas = np.random.multivariate_normal(np.zeros(D), Q, size=(N, 2))
        vel_samples = np.zeros((N, D))
        for j in range(N):
            x0 = vel * 0 + etas[j, 0]
            x1 = vel * dt + etas[j, 1]
            est_vel = (x1 - x0)/dt
            vel_samples[j, :] = est_vel
        Q_num = np.cov(vel_samples, rowvar=False)[np.newaxis][np.newaxis]
        results[i] = np.linalg.det(Q_num)**(1/D)
    return dts, results

def main():
    fig, ax = plt.subplots(2, figsize=(13,8))
    plt.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.1, hspace=0.3)



    T = 2.5
    # sigma = 10**(-7)
    sigma = 3
    vel_prior_sigma = 1
    triple_sigma = 10**(-7)
    dts, detQ, detQv = simulate_cov_growth(sigma, (1/30), T, triple_sigma, vel_prior_sigma)
    line_style = "-"
    alpha = 0.9
    ax[0].plot(dts, detQv, label="dt=1/30",  linewidth=2, color='r', linestyle=line_style, alpha=alpha)
    ax[1].plot(dts, detQ, label="dt=1/30", linewidth=2, color='r', linestyle=line_style, alpha=alpha)
    dts, detQ, detQv = simulate_cov_growth(sigma, (1/60), T, triple_sigma, vel_prior_sigma)
    ax[0].plot(dts, detQv, label="dt=1/60", linewidth=2, color='b', linestyle=line_style, alpha=alpha)
    ax[1].plot(dts, detQ, label="dt=1/60", linewidth=2, color='b', linestyle=line_style, alpha=alpha)
    dts, detQ, detQv = simulate_cov_growth(sigma, (1/90), T, triple_sigma, vel_prior_sigma)
    ax[0].plot(dts, detQv, label="dt=1/90", linewidth=2, color='g', linestyle=line_style, alpha=alpha)
    ax[1].plot(dts, detQ, label="dt=1/90", linewidth=2, color='g', linestyle=line_style, alpha=alpha)
    dts, detQ, detQv = simulate_cov_growth(sigma, (1/120), T, triple_sigma, vel_prior_sigma)
    ax[0].plot(dts, detQv, label="dt=1/120", linewidth=2, color='purple', linestyle=line_style, alpha=alpha)
    ax[1].plot(dts, detQ, label="dt=1/120", linewidth=2, color='purple', linestyle=line_style, alpha=alpha)
    dts, detQ, detQv = simulate_cov_growth(sigma, (1/150), T, triple_sigma, vel_prior_sigma)
    ax[0].plot(dts, detQv, label="dt=1/150", linewidth=2, color='orange', linestyle=line_style, alpha=alpha)
    ax[1].plot(dts, detQ, label="dt=1/150", linewidth=2, color='orange', linestyle=line_style, alpha=alpha)

    fontsize = 15

    ax[0].set_xlabel("elapsed time [s]", fontsize=fontsize)
    # ax[0].set_ylabel("|Q_vel|^(1/6)")
    ax[0].set_ylabel(r"$\sqrt[6]{|Q_{vel}|}$ [m/s]", fontsize=fontsize)



    ax[1].set_xlabel("elapsed time [s]", fontsize=fontsize)
    # ax[1].set_ylabel("|Q_pose|^(1/6)")
    ax[1].set_ylabel(r"$\sqrt[6]{|Q_{pose}|}$ [m]", fontsize=fontsize)
    ax[0].grid(alpha=0.5)
    ax[1].grid(alpha=0.5)

    #  predictions
    ax[0].plot(dts, dts * sigma + vel_prior_sigma, "-", label=f"analytical solution", linewidth=3, color='black')
    ax[1].plot(dts, sigma * dts**(3)/3 + vel_prior_sigma * dts**2, "-", label=f"analytical solution", linewidth=3, color='black')
    ax[0].set_title("Estimated velocity uncertainty", fontsize=fontsize)
    ax[1].set_title("Estimated pose uncertainty", fontsize=fontsize)

    ax[0].legend()
    ax[1].legend()

    sigma = 10**(-1)
    err_func1 = error_derivative_integration_so3r3_global
    err_func2 = error_derivative_integration_so3r3_global_new
    # ax[2].plot(*simulate_velocity_cov(err_func1, sigma), "-", label=f"cov_growth1", linewidth=4,color='blue')
    # # ax[1].plot(*simulate_velocity_cov(err_func2, sigma), "-", label=f"cov_growth1", linewidth=2,color='green')
    # ax[2].set_xlabel("dt")
    # ax[2].set_ylabel("|Qv|^(1/6)")
    # ax[2].grid()
    #
    # ax[2].plot(*simulate_num_vel_estimate(sigma), "-", label=f"cov_growth1", linewidth=2,color='red')
    # ax[2].set_xlabel("dt")
    # ax[2].set_ylabel("|Qv|^(1/6)")
    # ax[2].grid()
    fig.savefig(f"uncertainty.png")
    fig.savefig(f"uncertainty.svg")
    plt.show()
    # pass

if __name__ == "__main__":
    main()