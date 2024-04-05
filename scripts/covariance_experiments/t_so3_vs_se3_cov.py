import numpy as np
import pinocchio as pin
import pickle


# np.random.seed(2)
# pin.seed(0)
def random_cov(dim=6):
    A = np.random.rand(dim, dim)
    sig = 0.3  # meters and radians
    Q = sig**2*np.dot(A, A.T)
    Q[3:6, :3] = 0
    Q[:3, 3:6] = 0
    return Q

def sample_se3(T: pin.SE3, Q: np.ndarray):
    assert Q.shape[0] == Q.shape[1] == 6
    # Sample tangent space element in "local" frame ("right" convention)
    eta = np.random.multivariate_normal(np.zeros(6), Q)
    return T * pin.exp6(eta)

def load_Qs():
    with open('Q.p', 'rb') as file:
        Q = pickle.load(file)
    with open('Q_num.p', 'rb') as file:
        Q_num = pickle.load(file)
    with open('Q_num_txso3.p', 'rb') as file:
        Q_num_txso3 = pickle.load(file)
    return Q, Q_num, Q_num_txso3
def save_Qs(Q, Q_num, Q_num_txso3):
    with open('Q.p', 'wb') as file:
        pickle.dump(Q, file)
    with open('Q_num.p', 'wb') as file:
        pickle.dump(Q_num, file)
    with open('Q_num_txso3.p', 'wb') as file:
        pickle.dump(Q_num_txso3, file)

# Create random values for all transforms:
# T = pin.SE3.Random()    # estimated camera pose in world frame
def get_Qs():
    T = pin.SE3.Identity()
    Q = random_cov()


    # Verify numerically (Monte-Carlo)
    N_samples = int(2e4)  # no difference above
    nu_wb_arr = np.zeros((N_samples,6))
    print(f'Monte Carlo Sampling N_samples={N_samples}')
    for i in range(N_samples):
        if i % 4e4 == 0:
            print(f'{100*(i/N_samples)} %')
        T_n = sample_se3(T, Q)
        nu_wb = pin.log6(T_n)
        nu_wb_arr[i,:] = nu_wb.vector
    Q_num = np.cov(nu_wb_arr, rowvar=False)

    N_samples = int(4e4)  # no difference above
    nu_wb_arr = np.zeros((N_samples,6))
    print(f'Monte Carlo Sampling N_samples={N_samples}')
    for i in range(N_samples):
        if i % 1e4 == 0:
            print(f'{100*(i/N_samples)} %')
        T_n = sample_se3(T, Q)
        nu_wb_txso3 = np.zeros(6)
        nu_wb_txso3[:3] = T_n.translation
        nu_wb_txso3[3:6] = pin.log3(T_n.rotation)
        nu_wb_arr[i,:] = nu_wb_txso3
    Q_num_txso3 = np.cov(nu_wb_arr, rowvar=False)
    return Q, Q_num, Q_num_txso3

def frobenius_norm(Q1, Q2):
    # Distance measure between two matrices
    return np.sqrt(np.trace((Q1 - Q2).T @ (Q1 - Q2)))


Q, Q_num, Q_num_txso3 = get_Qs()
# Q, Q_num, Q_num_txso3 = load_Qs()
# save_Qs(Q, Q_num, Q_num_txso3)



print('Q_num')
print(frobenius_norm(Q_num, Q))
print('Q_num_txso3')
print(frobenius_norm(Q_num_txso3, Q))
print('')

