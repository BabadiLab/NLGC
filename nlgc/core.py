import numpy as np
import itertools
from scipy import linalg
from nlgc.opt.e_step import sskf
from nlgc.opt.m_step import solve_for_a, solve_for_q, calculate_ss, compute_ll
from nlgc.opt.opt import NeuraLVAR, NeuraLVARCV
from nlgc._utils import debiased_dev
from nlgc._stat import fdr_control
from nlgc._utils import bias, sample_path_bias, debiased_dev, mybias, my_debiased_dev
from matplotlib import pyplot as plt


def compare(i, model_f, model_r, x):
    fig, ax = plt.subplots()
    ax.plot(model_f._parameters[4][:, i], label='full')
    ax.plot(model_r._parameters[4][:, i], label='reduced')
    ax.plot(x[:, i], label='true')
    ax.legend()
    fig.show()
    return fig


def string_link(reg_idx, emod):
    temp = f"{list(range(reg_idx*emod, reg_idx*emod + emod))}"
    temp = temp.replace(" ", "")
    temp = temp.replace("[", "")
    temp = temp.replace("]", "")

    return temp


class GC:
    def __init__(self, subject, y, f, r, nx, ny, t, p, emod, d_raw, d, a_f, q_f, lambda_f):
        self.subject = subject
        self.y = y
        self.f = f
        self.r = r

        self.nx = nx
        self.ny = ny
        self.t = t
        self.p = p
        self.emod = emod
        self.d_raw = d_raw
        # scalar
        self.bias_f = bias_f
        # matrix
        self.bias_r = bias_r

        self.a_f = a_f
        self.q_f = q_f
        self.lambda_f = lambda_f

    def fdr(self, alpha):
        return fdr_control(self.d, self.p * self.emod, alpha)

    def save_(self, add):
        pass

    #plot


if __name__ == '__main__':
    n, nx, emod, p = 10, 3, 2, 3
    np.random.seed(0)

    m = nx * emod

    t = 500

    r = 0.1*np.eye(n)
    q = 0.1 * np.eye(m)
    q[0, 0] = 10
    q[1, 1] = 11
    q[2, 2] = 8
    # q[3, 3] = 9
    q[4, 4] = 11
    # q[5, 5] = 9.5
    # q[6, 6] = 10.3
    # q[7, 7] = 9.4
    # q[8, 8] = 10.1
    # q[9, 9] = 12.2

    a = np.zeros(p * m * m, dtype=np.float64)
    a.shape = (p, m, m)

    a[0, 0, 2] = -0.8

    a[0, 1, 2] = 0.6

    a[1, 1, 4] = -0.8

    # a[2, 2, 1] = 0.35

    # a[0, 3, 2] = -0.55

    # a[1, 3, 0] = 0.75

    # a[2, 4, 2] = 0.7

    # a[2, 3, 1] = -0.5

    # a[1, 5, 6] = 0.6
    #
    # a[0, 2, 8] = 0.8
    #
    # # a[1, 3, 8] = -0.5
    #
    # a[0, 8, 1] = -0.9
    #
    # a[1, 8, 4] = 0.55
    #
    # a[3, 4, 8] = -0.75
    #
    # a[2, 6, 9] = 0.85

    for i in range(1, m):
        a[2, i, i] = 0.1

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

    max_iter = 100
    max_cyclic_iter = 3
    tol = 1e-8
    kwargs = {'max_iter': max_iter,
              'max_cyclic_iter': max_cyclic_iter,
              'rel_tol': tol}

    # learn the full model
    model_f = NeuraLVARCV(p, 10, 5, 10, use_lapack=False)
    model_f.fit(y.T, f, r, lambda_range, a_init=None, q_init=np.eye(m), alpha=0.5, beta=0.1, **kwargs)
    a_f = model_f._parameters[0]
    q_f = model_f._parameters[2]
    x_f = model_f._parameters[4]
    lambda_f = model_f.lambda_
    bias_f = model_f.compute_bias(y.T)

    D = np.zeros((nx, nx))
    D_raw = np.zeros((nx, nx))

    # learn reduced models
    a_init = np.empty_like(a_f)

    q_d = np.diag(q_f)
    q_reg = [sum(q_d[i:i + 2]) for i in range(0, m, 2)]
    q_reg_sort = np.sort(q_reg)
    q_reg_sort = q_reg_sort[::-1]
    idx = np.argsort(q_reg)

    c = 0.9
    for i in range(nx):
        if np.sum(q_reg_sort[0:i]) > c*np.sum(q_reg_sort):
            break

    src_selection = idx[nx-i-1:nx]
    print(src_selection)
    for i, j in itertools.product(src_selection, repeat=2):
        if i == j:
            continue

        target = string_link(i, emod)
        src = string_link(j, emod)

        link = target+'->'+src
        print(link)
        a_init[:] = a_f[:]
        a_init[:, j*emod: (j+1)*emod, i*emod: (i+1)*emod] = 0
        model_r = NeuraLVAR(p, use_lapack=False)
        model_r.fit(y.T, f, r, lambda_f, a_init=a_init, q_init=q_f*1, restriction=link, alpha=0.5, beta=0.1, **kwargs)

        a_r = model_r._ravel_a(model_r._parameters[0])
        q_r = model_r._parameters[2]
        x_r = model_r._parameters[4]
        bias_r = model_r.compute_bias(y.T)

        D_raw[j, i] = -2 * (model_r._lls[-1] - model_f._lls[-1])
        D[j, i] = -2 * (model_r._lls[-1] - model_f._lls[-1]) - (bias_f - bias_r)


    alpha = 0.1
    J = fdr_control(D, p, alpha)
    plt.imshow(D[0:10, 0:10], vmin=0, vmax=150)
    plt.colorbar()
    plt.show()
    print(D)

