import numpy as np
import itertools
from scipy import linalg
from nlgc.opt.e_step import sskf
from nlgc.opt.m_step import solve_for_a, solve_for_q, calculate_ss, compute_ll
from nlgc.opt.opt import NeuraLVAR, NeuraLVARCV
from nlgc._utils import debiased_dev
from nlgc._stat import fdr_control
from nlgc._utils import debiased_dev, mybias, my_debiased_dev


    n, m, p = 3, 3, 2
def compare(i, model_f, model_r, x):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(model_f._parameters[4][:, i], label='full')
    ax.plot(model_r._parameters[4][:, i], label='reduced')
    ax.plot(x[:, i], label='true')
    ax.legend()
    fig.show()
    return fig

if __name__ == '__main__':
    np.random.seed(0)

    t = 1000

    r = 0.1*np.eye(n)
    q = 0.1 * np.eye(m)
    q[0, 0] = 10
    q[1, 1] = 8

    a = np.zeros(p * m * m, dtype=np.float64)
    a.shape = (p, m, m)

    a[0, 0, 0] = 0.5
    a[0, 0, 1] = -0.1
    a[0, 1, 1] = 0.9
    a[0, 2, 2] = 0.1

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
        x[i] = u[i]
        for k in range(p):
            x[i] += a[k].dot(x[i - k - 1])

    y = x.dot(f.T) + v

    # model parameters
    lambda_range = [1, 0.1, 0.01, 0.001]

    max_iter = 1000
    max_cyclic_iter = 2
    tol = 1e-8
    kwargs = {'max_iter': max_iter,
              'max_cyclic_iter': max_cyclic_iter,
              'rel_tol': tol}

    # learn the full model
    model_f = NeuraLVARCV(p, 10, 5, 10, use_lapack=False)
    model_f.fit(y.T, f, r, lambda_range, a_init=None, q_init=np.eye(m), **kwargs)
    a_f = model_f._parameters[0]
    q_f = model_f._parameters[2]
    x_f = model_f._parameters[4]
    lambda_f = model_f.lambda_

    D = np.zeros((m, m))
    D_ = np.zeros((m, m))
    D_prime = np.zeros((m, m))
    D_prime_ = np.zeros((m, m))

    # learn reduced models
    m=3
    a_init = np.empty_like(a_f)
    for i, j in itertools.product(range(m), repeat=2):
        if i == j:
            continue
        link = f"{i}->{j}"
        print(link)
        a_init[:] = a_f[:]
        a_init[:, j, i] = 0
        model_r = NeuraLVAR(p, use_lapack=False)
        model_r.fit(y.T, f, r, lambda_f, a_init=a_init, q_init=q_f*1, restriction=link, **kwargs)
        print(model_r._ravel_a(model_r._parameters[0]))

        a_r = model_r._parameters[0]
        a_r = np.hstack(a_r)

        q_r = model_r._parameters[2]
        x_r = model_r._parameters[4]
        compare(j, model_f, model_r, x)
        D[j, i] = debiased_dev(x_r[:, 0: m], x_f[:, 0: m], a_r, np.hstack(a_f), q_r, q_f, t, j, mo=1)
        D_[j, i] = my_debiased_dev(model_f, model_r, y.T, j)
        D_prime[j, i] = t * (np.log(q_r[j, j]) - np.log(q_f[j, j]))
        D_prime_[j, i] = -2*(model_r._lls[-1] - model_f._lls[-1])

    print(f"D:{D}")
    print(f"D_:{D_}")
    print(f"D_prime:{D_prime}")
    print(f"D_prime_:{D_prime_}")

alpha = 0.05
print(D)
J = fdr_control(D, p, alpha)
print(J)
