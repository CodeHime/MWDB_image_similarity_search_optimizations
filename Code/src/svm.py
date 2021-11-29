import cvxopt as optimizer
import cvxopt.solvers as solver
import numpy as np
from numpy import linalg
import os

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))


class SupportVectorMachine(object):

    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None:
            self.C = float(self.C)

    def fit(self, X, y):
        number_samples, number_features = X.shape

        K = np.zeros((number_samples, number_samples))

        for i in range(number_samples):
            for j in range(number_samples):
                K[i, j] = self.kernel(X[i], X[j])

        # Params initialization
        P = optimizer.matrix(np.outer(y, y) * K)
        q = optimizer.matrix(np.ones(number_samples) * -1)
        A = optimizer.matrix(y, (1, number_samples), 'd')
        b = optimizer.matrix(0.0)

        if self.C is None:
            G = optimizer.matrix(np.diag(np.ones(number_samples) * -1))
            h = optimizer.matrix(np.zeros(number_samples))
        else:
            tmp1 = np.diag(np.ones(number_samples) * -1)
            tmp2 = np.identity(number_samples)
            G = optimizer.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(number_samples)
            tmp2 = np.ones(number_samples) * self.C
            h = optimizer.matrix(np.hstack((tmp1, tmp2)))

        # solving the problem of maximization
        solution = solver.qp(P, q, G, h, A, b)

        a = np.ravel(solution['x'])

        # support vectors
        s_vec = a > 1e-5
        ind = np.arange(len(a))[s_vec]
        self.a = a[s_vec]
        self.s_vec = X[s_vec]
        self.s_vec_y = y[s_vec]

        self.b = 0
        for n in range(len(self.a)):
            self.b += self.s_vec_y[n]
            self.b -= np.sum(self.a * self.s_vec_y * K[ind[n], s_vec])
        if len(self.a) > 0:
            self.b /= len(self.a)
        if self.kernel == linear_kernel:
            self.w = np.zeros(number_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.s_vec_y[n] * self.s_vec[n]
        else:
            self.w = None

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, s_vec_y, s_vec in zip(self.a, self.s_vec_y, self.s_vec):
                    s += a * s_vec_y * self.kernel(X[i], s_vec)
                y_predict[i] = s
            return y_predict + self.b

    def predict_result(self, X):
        prediction_values = self.project(X)
        return prediction_values, np.sign(prediction_values)

    def predict(self, X):
        return np.sign(self.project(X))