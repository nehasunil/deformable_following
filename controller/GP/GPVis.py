import GPy
import numpy as np
from IPython.display import display
from load_data import loadall
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab
from sklearn.metrics import mean_squared_error, r2_score
import math

X, Y = loadall()

N = X.shape[0]
print(N)
idx = list(range(N))
random.seed(0)
random.shuffle(idx)

train_idx = idx[:int(N * 0.8)]
test_idx = idx[int(N * 0.8):]

X_train, Y_train = X[train_idx], Y[train_idx]
X_test, Y_test = X[test_idx], Y[test_idx]

kernel1 = GPy.kern.Matern32(input_dim=4,ARD=True,initialize=False)
m1 = GPy.models.SparseGPRegression(X_train, Y_train[:, 0].reshape(Y_train.shape[0], 1), kernel1, num_inducing=1000, initialize=False)
m1.update_model(False)
m1.initialize_parameter()
m1[:] = np.load('GP/6435/m1_m32_a_sparse_1000i_20.npy')
m1.update_model(True)
# m.initialize_parameter()
# mu,var = m.predict(X_test)

kernel2 = GPy.kern.Exponential(input_dim=4, ARD=True, initialize=False)
m2 = GPy.models.SparseGPRegression(X_train, Y_train[:, 1].reshape(Y_train.shape[0], 1), kernel2, num_inducing=1000, initialize=False)
m2.update_model(False)
m2.initialize_parameter()
m2[:] = np.load('GP/6435/m2_exp_a_sparse_1000i_20.npy')
m2.update_model(True)

kernel3 = GPy.kern.Matern52(input_dim=4, ARD=True, initialize=False)
m3 = GPy.models.SparseGPRegression(X_train, Y_train[:, 2].reshape(Y_train.shape[0], 1), kernel3, num_inducing=1000, initialize=False)
m3.update_model(False)
m3.initialize_parameter()
m3[:] = np.load('GP/6435/m3_exp_a_sparse_1000i_20.npy')
m3.update_model(True)

#
#
#
# GPy.plotting.change_plotting_library('matplotlib')
# m.plot()
# matplotlib.pylab.show(block=True)

# Linearization
# var = [m1.param_array[0], m2.param_array[0]]
# length = [m1.param_array[1], m2.param_array[1]]
n = X_train.shape[0]
m = X_train.shape[1] - 1

print(n, m)

# g = np.asarray(Y_train)
#
# # gp is the model
# def K_inv(gp):
#     return np.linalg.inv([[var[gp]**2 * math.exp(-np.linalg.norm(X_train[i,:]-X_train[j,:])**2/(2*length[gp]**2)) for j in range(n)] for i in range(n)])
#
# # j is index of state taking the derivative of (wrt), gp is the model
# def k_deriv(x, j, gp):
#     # return np.asarray([-(var[gp]**2 / length[gp]**2) * (x[j] - X_train[i,j]) * math.exp(-(x[j] - X_train[i,j])**2/(2*length[gp]**2)) for i in range(n)])
#     return np.asarray([-(var[gp] ** 2 / length[gp] ** 2) * (x[j] - X_train[i, j]) * math.exp(
#         -np.linalg.norm(x - X_train[i, :]) ** 2 / (2 * length[gp] ** 2)) for i in range(n)]).reshape(1, n)
#
# A = np.asarray([[np.matmul(np.matmul(k_deriv([[0], [0], [0]], j, i), K_inv(i)), g[:,i]) for j in range(m)] for i in range(m)]).reshape(2,2)
# B = np.asarray([np.matmul(np.matmul(k_deriv([[0], [0], [0]], 2, i), K_inv(i)), g[:,i]) for i in range(m)]).reshape((2,1))

# dx = [(X[:, i].max() - X[:, i].min())*.0001 for i in range(m + 1)]
# model = [m1, m2, m3]
# A = np.zeros((m, m))
#
#
# for i in range(m):
#     for j in range(m):
#         if j is 0:
#             dy_neg = np.asarray(model[i].predict(np.asarray([[-dx[j], 0, 0, 0]]))).reshape(1, 2)[0,0]
#             dy_pos = np.asarray(model[i].predict(np.asarray([[dx[j], 0, 0, 0]]))).reshape(1, 2)[0,0]
#         elif j is 1:
#             dy_neg = np.asarray(model[i].predict(np.asarray([[0, -dx[j], 0, 0]]))).reshape(1, 2)[0,0]
#             dy_pos = np.asarray(model[i].predict(np.asarray([[0, dx[j], 0, 0]]))).reshape(1, 2)[0,0]
#         else:
#             dy_neg = np.asarray(model[i].predict(np.asarray([[0, 0, -dx[j], 0]]))).reshape(1, 2)[0, 0]
#             dy_pos = np.asarray(model[i].predict(np.asarray([[0, 0, dx[j], 0]]))).reshape(1, 2)[0, 0]
#         A[i, j] = (dy_pos-dy_neg)/(2*dx[j])
# print("A: ", A)
# B = np.zeros((m, 1))
# for i in range(m):
#     dy_neg = np.asarray(model[i].predict(np.asarray([[0, 0, 0, -dx[j]]]))).reshape(1, 2)[0,0]
#     dy_pos = np.asarray(model[i].predict(np.asarray([[0, 0, 0, dx[j]]]))).reshape(1, 2)[0,0]
#     B[i, 0] = (dy_pos-dy_neg)/(2*dx[j])
# print("B:", B)
#
# def gp_lin(x):
#     return np.matmul(A, x[:-1]) + np.matmul(B, [x[-1]])
#
#
# Y_pred_lin = np.asarray([gp_lin(x) for x in X_test])
# print("*** Linearized GP model ***")
# # The mean squared error
# print("Mean squared error: %.2f"
#       % mean_squared_error(Y_test, Y_pred_lin))
# # Explained variance score: 1 is perfect prediction
# print('X_dot Variance score: %.2f' % r2_score(Y_test[:, 0], Y_pred_lin[:,0]))
# print('Theta_dot Variance score: %.2f' % r2_score(Y_test[:, 1], Y_pred_lin[:,1]))
# print('Alpha_dot Variance score: %.2f' % r2_score(Y_test[:, 2], Y_pred_lin[:,2]))
#

# # Y_pred_lin = m1.predict(X_test[1000:2000])
# plt.plot([-2, 2], [-2, 2], 'k--')
# plt.plot(Y_test[:1000]*1000, Y_pred_lin[:1000]*1000, '.')
# fig = plt.gcf()
# fig.set_size_inches(5, 5)
# plt.ylim(-1.5, 1.5)
# plt.xlim(-1.5, 1.5)
# plt.xticks(np.arange(-1, 1.2, step=1))
# plt.show()

def tv_linA(x):
    m = 3
    model = [m1, m2, m3]
    A = np.zeros((m, m))
    for i in range(m):
        grad = model[i].predictive_gradients(np.array([x]))
        for j in range(m):
            A[i][j] = grad[0][0][j]
    return A

def tv_linB(x):
    m = 3
    model = [m1, m2, m3]
    B = np.zeros((m, 1))
    for i in range(m):
        grad = model[i].predictive_gradients(np.array([x]))
        B[i, 0] = grad[0][0][3]
    return B


def tv_lin(x0, x):
    return np.matmul(tv_linA(x0), x[:-1]-x0[:-1]) + np.matmul(tv_linB(x0), [x[-1]-x0[-1]])

print("A: ", tv_linA([0,0,0,0]))
print("B: ", tv_linB([0,0,0,0]))
Y_pred_lin = np.asarray([tv_lin([0, 0, 0, 0], x) for x in X_test])
# Y_pred_lin = np.asarray([tv_lin(x, x) for x in X_test])
print("*** Linearized GP model ***")
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(Y_test, Y_pred_lin))
# Explained variance score: 1 is perfect prediction
print('X_dot Variance score: %.2f' % r2_score(Y_test[:, 0], Y_pred_lin[:, 0]))
print('Theta_dot Variance score: %.2f' % r2_score(Y_test[:, 1], Y_pred_lin[:, 1]))
print('Alpha_dot Variance score: %.2f' % r2_score(Y_test[:, 2], Y_pred_lin[:, 2]))
#
# Y_pred_lin = m1.predict(X_test[1000:2000])
# plt.plot([-2, 2], [-2, 2], 'k--')
# plt.plot(Y_test[1000:2000, 0] * 1000, Y_pred_lin * 1000, '.')
# fig = plt.gcf()
# fig.set_size_inches(5, 5)
# plt.ylim(-1.5, 1.5)
# plt.xlim(-1.5, 1.5)
# plt.xticks(np.arange(-1, 1.2, step=1))
# plt.show()