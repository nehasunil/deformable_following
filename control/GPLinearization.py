import GPy
import numpy as np
from IPython.display import display
from load_data import loadall
import random
from sklearn.metrics import mean_squared_error, r2_score
import math


# Setting up training and testing data
X, Y = loadall()

print(X[:, 0].max(), X[:, 0].min())

N = X.shape[0]
print(N)
idx = list(range(N))
random.seed(0)
random.shuffle(idx)

train_idx = idx[:int(N * 0.8)]
test_idx = idx[int(N * 0.8):]

X_train, Y_train = X[train_idx], Y[train_idx]
X_test, Y_test = X[test_idx], Y[test_idx]

X_train_mini, Y_train_mini = X_train[:int(N*0.1)], Y_train[:int(N*0.1)]

print(X_train.shape, Y_train.shape)
print(X_train_mini.shape, Y_train_mini.shape)



# GP
# Regression
#
# # Y_metadata = {'output_index':np.arange(Y_train_mini[:, 0].reshape(Y_train_mini.shape[0], 1).shape[0])[:,None]}
# # ep = GPy.inference.latent_function_inference.expectation_propagation.EP()
# # # likelihood = GPy.likelihoods.Gaussian()
# # likelihood = GPy.likelihoods.HeteroscedasticGaussian(Y_metadata=Y_metadata)
# # mini
# kernel1_mini = GPy.kern.RatQuad(input_dim=4, ARD=True) #, lengthscale=0.7663)
# kernel2_mini = GPy.kern.RatQuad(input_dim=4, ARD=True)# lengthscale=0.7832)
# kernel3_mini = GPy.kern.RatQuad(input_dim=4, ARD=True)#, lengthscale=0.1447)
# m1_mini = GPy.models.GPRegression(X_train_mini, Y_train_mini[:, 0].reshape(Y_train_mini.shape[0], 1), kernel1_mini)
# m2_mini = GPy.models.GPRegression(X_train_mini, Y_train_mini[:, 1].reshape(Y_train_mini.shape[0], 1), kernel2_mini)
# m3_mini = GPy.models.GPRegression(X_train_mini, Y_train_mini[:, 2].reshape(Y_train_mini.shape[0], 1), kernel3_mini)
# # m1_mini = GPy.core.GP(X_train_mini, Y_train_mini[:, 0].reshape(Y_train_mini.shape[0], 1), kernel=kernel1_mini, likelihood=likelihood, inference_method=ep)
# # m2_mini = GPy.core.GP(X_train_mini, Y_train_mini[:, 1].reshape(Y_train_mini.shape[0], 1), kernel=kernel2_mini, likelihood=likelihood, inference_method=ep)
# # m3_mini = GPy.core.GP(X_train_mini, Y_train_mini[:, 2].reshape(Y_train_mini.shape[0], 1), kernel=kernel3_mini, likelihood=likelihood, inference_method=ep)
#
#
# m1_mini.optimize(messages=True)
# m2_mini.optimize(messages=True)
# m3_mini.optimize(messages=True)
# #
# # np.save('GP/6435/GP_m1_exp_mini.npy', m1_mini.param_array)
# #
# # print("test shape: ", Y_test.shape)
#
# Y_pred_x = m1_mini.predict(X_test)
# Y_pred_theta = m2_mini.predict(X_test)
# Y_pred_alpha = m3_mini.predict(X_test)
# print("*** GP model ***")
# print("dataset: mini")
# print("Kernel: Best")
# # The mean squared error
# print("X_dot Mean squared error: {:.3e}".format(mean_squared_error(Y_test[:, 0], Y_pred_x[0][:,0])))
# print("Theta_dot Mean squared error: {:.3e}".format(mean_squared_error(Y_test[:, 1], Y_pred_theta[0][:,0])))
# print("Alpha_dot Mean squared error: {:.3e}".format(mean_squared_error(Y_test[:, 2], Y_pred_alpha[0][:,0])))
# # Explained variance score: 1 is perfect prediction
# print('X_dot Variance score: %.3f' % r2_score(Y_test[:, 0], Y_pred_x[0][:,0]))
# print('Theta_dot Variance score: %.3f' % r2_score(Y_test[:, 1], Y_pred_theta[0][:,0]))
# print('Alpha_dot Variance score: %.3f' % r2_score(Y_test[:, 2], Y_pred_alpha[0][:,0]))

# print("Model x_dot")
# display(m1_mini)
# print("Model theta_dot")
# display(m2_mini)
# print("Model alpha_dot")
# display(m3_mini)


#
# full
kernel1 = GPy.kern.Matern32(input_dim=4, ARD=True)
kernel2 = GPy.kern.Exponential(input_dim=4, ARD=True)
kernel3 = GPy.kern.Exponential(input_dim=4, ARD=True)
m1 = GPy.models.SparseGPRegression(X_train, Y_train[:, 0].reshape(Y_train.shape[0], 1), kernel1, num_inducing=2000)
m2 = GPy.models.SparseGPRegression(X_train, Y_train[:, 1].reshape(Y_train.shape[0], 1), kernel2, num_inducing=2000)
m3 = GPy.models.SparseGPRegression(X_train, Y_train[:, 2].reshape(Y_train.shape[0], 1), kernel3, num_inducing=2000)
# display(m)
# GPy.plotting.change_plotting_library('matplotlib')
# m.plot()
# matplotlib.pylab.show(block=True)

m1.optimize(messages=True)
#
Y_pred_x = m1.predict(X_test)
print("X_dot Mean squared error: {:.3e}".format(mean_squared_error(Y_test[:, 0], Y_pred_x[0][:,0])))
print('X_dot Variance score: %.3f' % r2_score(Y_test[:, 0], Y_pred_x[0][:,0]))
#
#
m2.optimize(messages=True)
Y_pred_theta = m2.predict(X_test)
print("Theta_dot Mean squared error: {:.3e}".format(mean_squared_error(Y_test[:, 1], Y_pred_theta[0][:,0])))
print('Theta_dot Variance score: %.3f' % r2_score(Y_test[:, 1], Y_pred_theta[0][:,0]))
# #
# #
m3.optimize(messages=True)
# # #
# # #
Y_pred_alpha = m3.predict(X_test)
print("*** GP model ***")
print("dataset: full")
print("Kernel: RBF")
# The mean squared error
print("X_dot Mean squared error: {:.3e}".format(mean_squared_error(Y_test[:, 0], Y_pred_x[0][:,0])))
print("Theta_dot Mean squared error: {:.3e}".format(mean_squared_error(Y_test[:, 1], Y_pred_theta[0][:,0])))
print("Alpha_dot Mean squared error: {:.3e}".format(mean_squared_error(Y_test[:, 2], Y_pred_alpha[0][:,0])))
# Explained variance score: 1 is perfect prediction
print('X_dot Variance score: %.3f' % r2_score(Y_test[:, 0], Y_pred_x[0][:,0]))
print('Theta_dot Variance score: %.3f' % r2_score(Y_test[:, 1], Y_pred_theta[0][:,0]))
print('Alpha_dot Variance score: %.3f' % r2_score(Y_test[:, 2], Y_pred_alpha[0][:,0]))

np.save('GP/6435/m1_m32_a_sparse_2000i_80.npy', m1.param_array)
np.save('GP/6435/m2_exp_a_sparse_2000i_80.npy', m2.param_array)
np.save('GP/6435/m3_exp_a_sparse_2000i_80.npy', m3.param_array)

#
# print("Model x_dot")
# display(m1)
# # print("Model theta_dot")
# # display(m2)
# # print("Model alpha_dot")
# # display(m3)


#
#
# # Linearization
# # var = [m1.param_array[0], m2.param_array[0]]
# # length = [m1.param_array[1], m2.param_array[1]]
# n = X_train.shape[0]
# m = X_train.shape[1] - 1
#
# # g = np.asarray(Y_train)
# #
# # # gp is the model
# # def K_inv(gp):
# #     return np.linalg.inv([[var[gp]**2 * math.exp(-np.linalg.norm(X_train[i,:]-X_train[j,:])**2/(2*length[gp]**2)) for j in range(n)] for i in range(n)])
# #
# # # j is index of state taking the derivative of (wrt), gp is the model
# # def k_deriv(x, j, gp):
# #     # return np.asarray([-(var[gp]**2 / length[gp]**2) * (x[j] - X_train[i,j]) * math.exp(-(x[j] - X_train[i,j])**2/(2*length[gp]**2)) for i in range(n)])
# #     return np.asarray([-(var[gp] ** 2 / length[gp] ** 2) * (x[j] - X_train[i, j]) * math.exp(
# #         -np.linalg.norm(x - X_train[i, :]) ** 2 / (2 * length[gp] ** 2)) for i in range(n)]).reshape(1, n)
# #
# # A = np.asarray([[np.matmul(np.matmul(k_deriv([[0], [0], [0]], j, i), K_inv(i)), g[:,i]) for j in range(m)] for i in range(m)]).reshape(2,2)
# # B = np.asarray([np.matmul(np.matmul(k_deriv([[0], [0], [0]], 2, i), K_inv(i)), g[:,i]) for i in range(m)]).reshape((2,1))
#
# dx = [(X[:, i].max() - X[:, i].min())*.0001 for i in range(m + 1)]
# model = [m1, m2, m3]
# A = np.zeros((m, m))
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