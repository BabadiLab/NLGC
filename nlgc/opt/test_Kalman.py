import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt
from nlgc.opt.e_step import sskf
from nlgc.opt.m_step import solve_for_a, solve_for_q, calculate_ss, compute_ll
from nlgc.opt.opt import NeuraLVAR, NeuraLVARCV

# np.random.seed(5)

# n, m, p, k = 155, 2*68, 6, 100
n, m, p, k = 4, 3, 2, 4
t = 1000
q = np.eye(m)
r = np.eye(n)

sn = np.random.standard_normal((m + n) * t)
u = sn[:m * t]
u.shape = (t, m)
l = linalg.cholesky(q, lower=True)
u = u.dot(l.T)
v = sn[m * t:]
v.shape = (t, n)
l = linalg.cholesky(r, lower=True)
v = v.dot(l.T)
a = np.zeros(p * m * m, dtype=np.float64)
for i, val in zip(np.random.choice(p * m * m, k), np.random.randn(k)):
    a[i] = val
a.shape = (p, m, m)
a[0] /= 1.1 * linalg.norm(a[0])
a[1] /= 1.1 * linalg.norm(a[1])
f = np.random.randn(n, m)
x = np.empty((t, m), dtype=np.float64)
x[0] = 0.0
x[1] = 0.0
for x_, _x, __x, u_ in zip(x[2:], x[1:], x, u):
    x_[:] = a[0].dot(_x) + a[1].dot(__x) + u_
y = x.dot(f.T) + v
fig, ax = plt.subplots()
ax.plot(x)
ax.plot(y)
fig.show()
print(l)

def test_neuralvar(use_lapack=True, lambda2=0.1, rel_tol=0.0001):
    model = NeuraLVAR(p, use_lapack=use_lapack)
    y_train = y[:900]
    y_test = y[900:]
    a_, q_upper, lls, *rest = \
        model._fit(y_train.T, f, r, lambda2=lambda2, max_iter=100, max_cyclic_iter=2, a_init=None, q_init=0.1*q,
        rel_tol=rel_tol)
    print(a_)
    print('\n')
    print(a)
    print(q_upper)
    print(lls)
    print(model.compute_ll(y_test.T, (a_, f, q_upper, r)))
    model = NeuraLVAR(p, use_lapack=use_lapack)
    model.fit(y_train.T, f, r, lambda2=lambda2, max_iter=100, max_cyclic_iter=2, a_init=None, q_init=0.1*q, rel_tol=rel_tol)
    print(f"ll:{model.compute_ll(y_test.T, None)}")
    model = NeuraLVAR(p, use_lapack=use_lapack)
    model.fit(y_train.T, f, r, lambda2=lambda2, max_iter=100, max_cyclic_iter=2, a_init=None, q_init=0.1 * q,
              rel_tol=rel_tol, restriction='1->2')
    print(f"ll:{model.compute_ll(y_test.T, None)}")
    model = NeuraLVAR(p, use_lapack=use_lapack)
    model.fit(y.T, f, r, lambda2=lambda2, max_iter=100, max_cyclic_iter=2, a_init=None, q_init=0.1 * q,
              rel_tol=rel_tol)
    print(model._parameters)


def test_neuralvarcv(use_lapack=True, lambda2=0.05, rel_tol=0.0001):
    model = NeuraLVARCV(p, 10, 5, 10, use_lapack=use_lapack)
                    # order, max_n_mus, cv, n_jobs,
    model.fit(y.T, f, r, lambda_range=[0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001], max_iter=100, max_cyclic_iter=2, a_init=None,
              q_init=0.1*q, rel_tol=rel_tol)
    print(a)
    print('\n')
    print(model._parameters[0])
    print('\n')
    print(model._parameters[2])
    print(model.lambda_)


def test_em(use_lapack=True, lambda2=0.05):
    # pr = cProfile.Profile()
    # pr.enable()
    # Initialization
    a_upper = np.zeros((m, m*p))
    a_upper[:] = 0.0*np.reshape(np.swapaxes(a, 0, 1), (m, m*p))
    q_upper = 0.1*q.copy()
    _x = np.empty((y.shape[0], m*p), dtype=np.float64)
    x_ = np.empty_like(_x)

    # Augmented states
    a_upper = np.zeros((m, m*p))
    a_lower = np.hstack((np.eye(m*(p-1)), np.zeros((m*(p-1), m))))
    a_ = np.vstack((a_upper, a_lower))
    q_ = np.zeros((m*p, m*p))
    non_zero_indices = np.diag_indices_from(q)
    f_ = np.hstack((f, np.zeros((n, m*(p-1)))))

    lls = []
    for _ in range(20):
        a_[:m] = a_upper
        q_[non_zero_indices] = q_upper[non_zero_indices]

        x_, s_, b, s_hat = sskf(y, a_, f_, q_, r, xs=(_x, x_), use_lapack=use_lapack)
        ll = compute_ll(y, x_, s_, s_hat, a_upper, f, q_upper, r, m, n, p)
        lls.append(ll)

        s1, s2, s3 = calculate_ss(x_, s_, b, m, p)

        # x_bar = np.empty((t-1, m*p), dtype=np.float64)
        # for _x, __x, x_ in zip(x[1:], x, x_bar):
        #     x_[:m] = _x
        #     x_[m:] = __x
        # s1 = x[2:].T.dot(x_bar[:-1]) / (x_bar.shape[0] - p + 1)
        # s2 = x_bar[:-1].T.dot(x_bar[:-1]) / (x_bar.shape[0] - p + 1)
        # s3 = x[2:].T.dot(x[2:]) / (x_bar.shape[0] - p + 1)

        # a_upper = np.zeros((m, m*p))
        # a_upper[:] = 0*np.reshape(np.swapaxes(a, 0, 1), (m, m*p))
        # q_upper = 0.1*q.copy()
        for _ in range(2):
            a_upper, changes = solve_for_a(q_upper, s1, s2, a_upper, lambda2=lambda2, max_iter=1000, tol=0.01)
            q_upper = solve_for_q(q_upper, s3, s1, a_upper, lambda2=lambda2)

    # pr.disable()
    # s1 = io.StringIO()
    # ps = pstats.Stats(pr, stream=s1).sort_stats(pstats.SortKey.CUMULATIVE)
    # ps.print_stats()
    # print(s1.getvalue())
    a_upper = np.swapaxes(np.reshape(a_upper, (m, p, m)), 0, 1)
    return a_upper, q_upper, lls


def inspect_results(use_lapack=True, lambda2=0.05):
    a_upper, q_upper, lls = test_em(use_lapack=use_lapack, lambda2=lambda2)
    print(a_upper)
    print('\n')
    print(a)
    print(q_upper)
    print(lls)


if __name__ == '__main__':
    test_neuralvar()
    test_neuralvarcv(rel_tol=0.00001)
    # import timeit
    # print("use_lapack=False")
    # print(timeit.repeat("test_em(use_lapack=False)", setup="from __main__ import test_em", number=1, repeat=5))
    # print("use_lapack=True")
    # print(timeit.repeat("test_em(use_lapack=True)", setup="from __main__ import test_em", number=1, repeat=5))
