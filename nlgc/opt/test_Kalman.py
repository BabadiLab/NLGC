import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt
from nlgc.opt.e_step import sskf
from nlgc.opt.m_step import solve_for_a, solve_for_q, calculate_ss, compute_ll
from nlgc.opt.opt import NeuraLVAR, NeuraLVARCV


def generate_processes(seed=0):
    np.random.seed(seed)
    # n, m, p, k = 155, 2*68, 6, 100
    n, m, p, k = 3, 3, 2, 4
    n_eigenmodes = 2
    t = 1000
    q = np.eye(m)
    # q[2, 2] = 0.001
    r = 1 * np.eye(n)

    sn = np.random.standard_normal((m + n) * t)
    u = sn[:m * t]
    u.shape = (t, m)
    l = linalg.cholesky(q, lower=True)
    u = u.dot(l.T)
    v = sn[m * t:]
    v.shape = (t, n)
    l = linalg.cholesky(r, lower=True)
    v = v.dot(l.T)

    # -------------------------------------------------------------------
    # a = np.zeros(p * m * m, dtype=np.float64)
    # for i, val in zip(np.random.choice(p * m * m, k), np.random.randn(k)):
    #     a[i] = val
    # a.shape = (p, m, m)
    # a[0] /= 1.1 * linalg.norm(a[0])
    # a[1] /= 1.1 * linalg.norm(a[1])
    # a[1, 1, 1] = - 0.98

    a = np.zeros(p * m * m, dtype=np.float64)
    a.shape = (p, m, m)

    a[0, 0, 1] = -0.1

    a[0, 0, 0] = 0.98

    a[1, 1, 1] = 0.985

    a[0, 2, 2] = 0.5
    print(a)
    # --------------------------------------------------------------------

    f = np.random.randn(n, m)
    f /= np.sqrt(f ** 2).sum(axis=0)
    # f = np.eye(n)
    x = np.empty((t, m), dtype=np.float64)
    x[0] = 0.0
    x[1] = 0.0
    for x_, _x, __x, u_ in zip(x[2:], x[1:], x, u):
        x_[:] = a[0].dot(_x) + a[1].dot(__x) + u_

    y_ = x.dot(f.T)
    y = 10 * np.sqrt((v**2).sum() / (y_**2).sum()) * y_ + v
    fig, ax = plt.subplots(2)
    ax[0].plot(x)
    ax[1].plot(y)
    fig.show()
    return x, y, f, r, p, a, q


def test_neuralvar(use_lapack=True, lambda2=0.1, rel_tol=0.0001):
    x, y, f, r, p, a, q = generate_processes(0)
    model = NeuraLVAR(p, use_lapack=use_lapack)
    y_train = y[:900]
    y_test = y[900:]
    model = model.fit(y_train.T, f, r, lambda2=lambda2, max_iter=200, max_cyclic_iter=2, a_init=a.copy()*0,
                      q_init=0.1*q, rel_tol=rel_tol)
    # print(a_)
    # print('\n')
    # print(a)
    # print(q_upper)
    # print(lls)
    # print(model.compute_ll(y_test.T, (a_, f, q_upper, r)))
    # model = NeuraLVAR(p, use_lapack=use_lapack)
    # model.fit(y_train.T, f, r, lambda2=lambda2, max_iter=100, max_cyclic_iter=2, a_init=None, q_init=0.1*q, rel_tol=rel_tol)
    # print(f"ll:{model.compute_ll(y_test.T, None)}")
    # model = NeuraLVAR(p, use_lapack=use_lapack)
    # model.fit(y_train.T, f, r, lambda2=lambda2, max_iter=100, max_cyclic_iter=2, a_init=None, q_init=0.1 * q,
    #           rel_tol=rel_tol, restriction='1->2')
    # print(f"ll:{model.compute_ll(y_test.T, None)}")
    # model = NeuraLVAR(p, use_lapack=use_lapack)
    # model.fit(y.T, f, r, lambda2=lambda2, max_iter=100, max_cyclic_iter=2, a_init=None, q_init=0.1 * q,
    #           rel_tol=rel_tol)
    return model


def test_neuralvarcv(use_lapack=True, lambda2=0.05, rel_tol=0.0001, seed=0):
    x, y, f, r, p, a, q = generate_processes(seed)
    model = NeuraLVARCV(p, p, 1, 10, 3, 10, use_lapack=use_lapack)
                    # order, max_n_mus, cv, n_jobs,
    model.fit(y.T, f, r, lambda_range=[1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001], max_iter=100,
              max_cyclic_iter=2, a_init=None,
              q_init=0.1*q, rel_tol=rel_tol)
    print(a)
    print('\n')
    print(model._parameters[0])
    print('\n')
    # print(model._parameters[2])
    # print(model.lambda_)
    return model


def test_nlgc(use_lapack=True, max_iter=1000, lambda2=None, rel_tol=1e-6, use_es=False, seed=0):
    x, y, f, r, p, a, q = generate_processes(seed)
    epsilon = 0.02
    alpha = 2 + epsilon
    beta = 1 * (1 + epsilon)
    from nlgc._nlgc import _gc_extraction
    out = _gc_extraction(y.T, f, r, p, p, n_eigenmodes=1, ROIs=[0, 1, 2], alpha=alpha, beta=beta,
                  lambda_range=lambda2, max_iter=max_iter, max_cyclic_iter=3,
                  tol=rel_tol, sparsity_factor=0.0, cv=5, use_lapack=use_lapack, use_es=use_es)
    return out


def inspect_results(use_lapack=True, lambda2=0.05):
    a_upper, q_upper, lls = test_em(use_lapack=use_lapack, lambda2=lambda2)
    print(a_upper)
    print('\n')
    print(a)
    print(q_upper)
    print(lls)


if __name__ == '__main__':
    # test_neuralvar()
    # test_neuralvarcv(rel_tol=0.00001)
    # import timeit
    print("use_lapack=False")
    # print(timeit.repeat("test_em(use_lapack=False)", setup="from __main__ import test_em", number=1, repeat=5))
    # print("use_lapack=True")
    # print(timeit.repeat("test_em(use_lapack=True)", setup="from __main__ import test_em", number=1, repeat=5))
