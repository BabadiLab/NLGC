import numpy as np
import ipdb
import matplotlib
import matplotlib.pyplot as plt
from scipy import linalg


def source_estimation(y, f, r, eta=1):
    w, v = np.linalg.eigh(r)
    r_ = v.dot(np.diag(np.sqrt(w))).dot(v.T)

    f_ = np.linalg.solve(r_, f)
    y_ = np.linalg.solve(r_, y)

    d, u = np.linalg.eigh(f_.dot(f.T))

    z = u.T.dot(y_)

    ny, t = y.shape
    _, nx = f.shape

    lambda_ = 0
    for i in range(1000):
        lambda_ -= (func(z, d, lambda_) - eta*ny*t) / func_dot(z, d, lambda_)

    x = np.linalg.solve(np.eye(nx)/lambda_ + f_.T.dot(f_), f_.T.dot(y_))

    return x.T


def func(z, d, x):
    s = 0
    for i in range(len(d)):
        s += np.sum(z[i, :]**2) / ((1+d[i]*x)**2)

    return s


def func_dot(z, d, x):
    s = 0
    for i in range(len(d)):
        s -= 2*d[i]*np.sum(z[i, :]**2) / ((1+d[i]*x)**3)

    return s


def param_estimation(x, p, restriction=None):

    t, nx = x.shape

    X = np.zeros((t - p, p*nx))

    for k in range(p):
        X[:, k * nx:(k + 1) * nx] = x[p - 1 - k:t - 1 - k]

    # add shrinkage and proximal***

    a = np.linalg.solve(X.T.dot(X), X.T.dot(x[p:, :]))
    q = np.diag(np.sum((x[p:]-X.dot(a))**2, axis=0)/(t-p))

    #change a to (p,nx,nx) form

    return a, q

if __name__ == "__main__":

    t = 100
    ny = 2
    nx = 4

    r = 1 * np.eye(ny)
    f = np.random.randn(ny, nx)
    x = 10 * np.random.randn(t, nx)

    u = np.random.standard_normal(ny * t)
    u.shape = (t, ny)
    l = linalg.cholesky(r, lower=True)
    u = u.dot(l.T)

    y = x.dot(f.T) + u

    x_ = source_estimation(y.T, f, r)

    a, q = param_estimation(x_, 1)

    # fig, ax = plt.subplots(2)
    #
    # ax[0].plot(x)
    # ax[1].plot(x_)
    #
    # fig.show()
