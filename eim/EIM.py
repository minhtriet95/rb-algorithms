import numpy as np
import matplotlib.pyplot as plt

N_x = 5
# Spatial domain
x1 = np.linspace(0, 1, num=N_x)
x2 = np.linspace(0, 1, num=N_x)
x_coord = np.array(np.meshgrid(x1, x2))
x_coord = x_coord.reshape(2, x1.size**2)

N_mu = 6
# Parameter samples
mu1 = np.linspace(-1, -0.01, num=N_mu)
mu2 = np.linspace(-1, -0.01, num=N_mu)
mu_sample = np.array(np.meshgrid(mu1, mu2)).reshape(2, mu1.size**2)


def G(x, mu): return 1.0/((x[0] - mu[0]) ** 2 + (x[1] - mu[1]) ** 2) ** 0.5
def G_exact(mu): return G(x_coord, mu)


G_mu_exact = np.zeros((mu1.size**2, x1.size**2))
for i in range(len(mu_sample[0])):
    mu_row = [mu_sample[0][i], mu_sample[1][i]]
    G_mu_exact[i] = G_exact(mu_row)

G_mu_EIM = np.zeros((mu1.size**2, x1.size**2))

# EIM offline process
q, q_max = 1, 50
converged = False
H = np.zeros((1, x1.size**2))
T = np.zeros((1, 2))
index_x_set = []
while not converged:
    # 1. Pick sample points
    index_mu = np.argmax(np.linalg.norm(
        G_mu_exact - G_mu_EIM, ord=np.inf, axis=1))
    mu_q = [mu_sample[0][index_mu], mu_sample[1][index_mu]]

    # 2. Find interpolation points
    xi_q_exact = G_mu_exact[:][index_mu]
    xi_q_EIM = G_mu_EIM[:][index_mu]
    residual = xi_q_exact - xi_q_EIM
    index_x = np.argmax(abs(residual))
    t_q = [x_coord[0][index_x], x_coord[1][index_x]]

    # 3. Find basis function as scaled errors
    h_q = residual/residual[index_x]

    # 4. Check convergence
    eim_error = np.linalg.norm(residual, ord=np.inf)
    if eim_error <= 1E-5 or q >= q_max:
        converged = True

    # 5. Update spaces
    if q == 1:
        H = np.array([h_q])
    else:
        H = np.concatenate((H, [h_q]), axis=0)
    T = np.concatenate((T, [t_q]), axis=0)
    index_x_set = np.concatenate((index_x_set, [index_x]), axis=0)

    B = np.zeros((q, q))
    for basis in range(q):
        for interpolant in range(q):
            B[basis][interpolant] = H[basis][int(index_x_set[interpolant])]

    G_mu_ti = np.zeros((len(index_x_set), 1))
    for i in range(mu1.size**2):
        G_mu_i = G_mu_exact[i]
        for index, value in enumerate(index_x_set):
            G_mu_ti[index] = G_mu_i[int(value)]
        c = np.linalg.solve(B.transpose(), G_mu_ti)
        G_mu_EIM_i = np.matmul(H.transpose(), c)
        G_mu_EIM[i][:] = G_mu_EIM_i.transpose()
    
    print('q = {}\t error = {error:e}'.format(q, error=eim_error))

    q = q + 1

# print(B)