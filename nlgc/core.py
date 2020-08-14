import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt
from nlgc.opt.e_step import sskf
from nlgc.opt.m_step import solve_for_a, solve_for_q, calculate_ss, compute_ll
from nlgc.opt.opt import NeuraLVAR, NeuraLVARCV
from nlgc._utils import debiased_dev

if __name__ == '__main__':

    n, m, p = 3, 3, 1
    t = 1000

    r = np.eye(n)
    q = 0.1 * np.eye(m)
    q[0, 0] = 10
    q[1, 1] = 8

    a = np.zeros(p * m * m, dtype=np.float64)
    a.shape = (p, m, m)

    a[0, 0, 0] = 0.5
    a[0, 0, 1] = -0.6
    a[0, 1, 1] = 0.9

    sn = np.random.standard_normal((m + n) * t)
    u = sn[:m * t]
    u.shape = (t, m)
    l = linalg.cholesky(q, lower=True)
    u = u.dot(l.T)
    v = sn[m * t:]
    v.shape = (t, n)
    l = linalg.cholesky(r, lower=True)
    v = v.dot(l.T)

    f = np.random.randn(n, m)
    x = np.empty((t, m), dtype=np.float64)
    for i in range(p):
        x[i] = 0.0

    for i in range(p, t):
        x[i, :] = u[i, :]
        for k in range(p):
            x[i, :] += a[k, :, :].dot(x[i - k - 1, :])
    # for x_, _x, __x, u_ in zip(x[2:], x[1:], x, u):
    #     x_[:] = a[0].dot(_x) + a[1].dot(__x) + u_
    y = x.dot(f.T) + v

    # model parameters
    lambda_range = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]

    max_iter = 100
    max_cyclic_iter = 3
    tol = 0.00001

    # learn the full model
    model_f = NeuraLVARCV(p, 10, 5, 10, use_lapack=True)
    model_f.fit(y.T, f, r, lambda_range, max_iter, max_cyclic_iter, a_init=None, q_init=np.eye(m), rel_tol=tol)
    a_f = model_f._parameters[0]
    q_f = model_f._parameters[2]
    x_f = model_f._parameters[4]
    lambda_f = model_f.lambda_

    D = np.zeros((m, m))
    # learn reduced models
    model_r = NeuraLVAR(p, use_lapack=True)
    m=1
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            link = str(i) + '->' + str(j)
            print(link)
            a_init = a_f
            a_init[:, j, i] = 0
            model_r.fit(y.T, f, r, lambda_f, max_iter, max_cyclic_iter, a_init, q_init=q_f, rel_tol=tol, restriction=link)
            print(model_r._ravel_a(model_r._parameters[0]))

            a_r = model_r._parameters[0]
            a_r = np.hstack(a_r)

            q_r = model_r._parameters[2]
            x_r = model_r._parameters[4]
            D[j, i] = debiased_dev(x_r[:, 0: m], x_f[:, 0: m], a_r, np.hstack(a_f), q_r, q_f, t, j, mo=1)

    print(D)