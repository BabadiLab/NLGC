 # Author: Behrad Soleimani <behrad@umd.edu>

import ipdb
import numpy as np
import itertools
from scipy import linalg
from nlgc.opt.e_step import sskf
from nlgc.opt.m_step import solve_for_a, solve_for_q, calculate_ss, compute_ll
from nlgc.opt.opt import NeuraLVAR, NeuraLVARCV
from nlgc._utils import debiased_dev
from nlgc._stat import fdr_control
from matplotlib import pyplot as plt
from tqdm import tqdm

def string_link(reg_idx, emod):
    temp = f"{list(range(reg_idx*emod, reg_idx*emod + emod))}"
    temp = temp.replace(" ", "")
    temp = temp.replace("[", "")
    temp = temp.replace("]", "")

    return temp


def gc_extraction(y, f, p, n_eigenmodes, ROIs = None, beta=0.1):

    n, m = f.shape
    nx = m // n_eigenmodes

    r = np.eye(n)

    lambda_range = [1, 0.1, 0.01, 0.001]

    max_iter = 50
    max_cyclic_iter = 5
    tol = 1e-8
    kwargs = {'max_iter': max_iter,
              'max_cyclic_iter': max_cyclic_iter,
              'rel_tol': tol}

    # learn the full model
    model_f = NeuraLVARCV(p, 10, 5, 10, use_lapack=False)

    alpha = 0.5

    model_f.fit(y.T, f, r, lambda_range, a_init=None, q_init=np.eye(m), alpha=alpha, beta=beta, **kwargs)

    a_f = model_f._parameters[0]
    q_f = model_f._parameters[2]
    # x_f = model_f._parameters[4]
    lambda_f = model_f.lambda_
    bias_f = model_f.compute_bias(y.T)

    dev_raw = np.zeros((nx, nx))
    bias_r = np.zeros((nx, nx))


    # learn reduced models
    a_init = np.empty_like(a_f)

    # q_d = np.diag(q_f)
    # q_reg = [sum(q_d[i:i + n_eigenmodes]) for i in range(0, m, n_eigenmodes)]
    # q_reg_sort = np.sort(q_reg)
    # q_reg_sort = q_reg_sort[::-1]
    # idx = np.argsort(q_reg)
    #
    # c = 0.9
    # for i in range(nx):
    #     if np.sum(q_reg_sort[0:i]) > c * np.sum(q_reg_sort):
    #         break
    #
    # src_selection = idx[nx - i - 1:nx]
    # print(src_selection)

    # ipdb.set_trace()
    # if ROIs == None:
    #     src_selection = range(0, nx)
    # else:
    #     src_selection = ROIs
    if ROIs is None:
        ROIs = range(0, nx)
    elif ROIs == 'just_full_model':
        ROIs = [0]

    sparsity = np.sum(np.absolute(a_f), axis=0)
    sparsity_factor = 0.1

    for i, j in tqdm(itertools.product(ROIs, repeat=2)):
        if i == j:
            continue
        if np.sum(sparsity[j * n_eigenmodes: (j + 1) * n_eigenmodes, i * n_eigenmodes: (i + 1) * n_eigenmodes]) \
            <= sparsity_factor*np.max(a_f[:, j * n_eigenmodes: (j + 1) * n_eigenmodes, :]):
            # print('It is sparse!')
            continue

        target = string_link(i, n_eigenmodes)
        src = string_link(j, n_eigenmodes)

        link = f"{target}->{src}"
        # "{:s}->{:s}".format(target, src)
        print(link)
        a_init[:] = a_f[:]
        a_init[:, j * n_eigenmodes: (j + 1) * n_eigenmodes, i * n_eigenmodes: (i + 1) * n_eigenmodes] = 0
        model_r = NeuraLVAR(p, use_lapack=False)
        model_r.fit(y.T, f, r, lambda_f, a_init=a_init, q_init=q_f * 1, restriction=link, alpha=alpha, beta=beta, **kwargs)

        # a_r = model_r._ravel_a(model_r._parameters[0])
        # q_r = model_r._parameters[2]
        # x_r = model_r._parameters[4]
        bias_r[j, i] = model_r.compute_bias(y.T)

        # if model_r._lls[-1] - model_f._lls[-1] > 0:
        #     continue
        # else:
        dev_raw[j, i] = -2 * (model_r._lls[-1] - model_f._lls[-1])

    return dev_raw, bias_r, bias_f, a_f, q_f, lambda_f, model_f._lls[-1]


def full_model_estimation(y, f, p, n_eigenmodes, beta=0.1):

    n, m = f.shape
    nx = m // n_eigenmodes

    r = np.eye(n)

    lambda_range = [1, 0.1, 0.01, 0.001]

    max_iter = 100
    max_cyclic_iter = 3
    tol = 1e-8
    kwargs = {'max_iter': max_iter,
              'max_cyclic_iter': max_cyclic_iter,
              'rel_tol': tol}

    # learn the full model
    model_f = NeuraLVARCV(p, 10, 5, 10, use_lapack=False)

    model_f.fit(y.T, f, r, lambda_range, a_init=None, q_init=np.eye(m), alpha=0.5, beta=beta, **kwargs)

    a_f = model_f._parameters[0]
    q_f = model_f._parameters[2]
    # x_f = model_f._parameters[4]
    # lambda_f = model_f.lambda_

    return model_f._lls[-1], a_f, q_f


def simulation_rnd():

    n, nx, n_eigenmodes, p = 10, 3, 1, 3
    np.random.seed(0)

    m = nx * n_eigenmodes

    t = 500

    r = np.eye(n)
    q = 0.1 * np.eye(m)
    q[0, 0] = 10
    q[1, 1] = 11
    q[2, 2] = 8
    # q[3, 3] = 9
    # q[4, 4] = 11
    # q[5, 5] = 9.5
    # q[6, 6] = 10.3
    # q[7, 7] = 9.4
    # q[8, 8] = 10.1
    # q[9, 9] = 12.2

    a = np.zeros(p * m * m, dtype=np.float64)
    a.shape = (p, m, m)

    a[0, 0, 2] = -0.8

    a[0, 1, 2] = 0.6

    # a[1, 1, 4] = -0.8

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

    return y, f, p, n_eigenmodes


class NLGC:
    def __init__(self, subject, nx, ny, t, p, n_eigenmodes, d_raw, bias_f, bias_r, a_f, q_f, lambda_f):

        self.subject = subject
        self.nx = nx
        self.ny = ny
        self.t = t
        self.p = p
        self.n_eigenmodes = n_eigenmodes
        self.d_raw = d_raw
        self.bias_f = bias_f
        self.bias_r = bias_r
        self.a_f = a_f
        self.q_f = q_f
        self.lambda_f = lambda_f

    def compute_debiased_dev(self):
        bias_mat = self.bias_r.copy()
        bias_mat[bias_mat != 0] -= self.bias_f
        # ipdb.set_trace()

        d = self.d_raw + bias_mat
        d[d <= 0] = 0
        np.fill_diagonal(d, 0)
        return d

    def fdr(self, alpha=0.05):
        return fdr_control(self.compute_debiased_dev(), self.p * self.n_eigenmodes, alpha)

    def save_(self, add):
        pass

    def plot(self, add):
        pass


if __name__ == '__main__':

    y, f, p, n_eigenmodes = simulation_rnd()

    d_raw, bias_r, bias_f, a_f, q_f = gc_extraction(y, f, p, n_eigenmodes)


    print(d_raw)

    # # model parameters
    # lambda_range = [1, 0.1, 0.01, 0.001]
    #
    # max_iter = 100
    # max_cyclic_iter = 3
    # tol = 1e-8
    # kwargs = {'max_iter': max_iter,
    #           'max_cyclic_iter': max_cyclic_iter,
    #           'rel_tol': tol}
    #
    # # learn the full model
    # model_f = NeuraLVARCV(p, 10, 5, 10, use_lapack=False)
    # model_f.fit(y.T, f, r, lambda_range, a_init=None, q_init=np.eye(m), alpha=0.5, beta=0.1, **kwargs)
    # a_f = model_f._parameters[0]
    # q_f = model_f._parameters[2]
    # x_f = model_f._parameters[4]
    # lambda_f = model_f.lambda_
    # bias_f = model_f.compute_bias(y.T)
    #
    # D = np.zeros((nx, nx))
    # D_raw = np.zeros((nx, nx))
    #
    # # learn reduced models
    # a_init = np.empty_like(a_f)
    #
    # q_d = np.diag(q_f)
    # q_reg = [sum(q_d[i:i + 2]) for i in range(0, m, 2)]
    # q_reg_sort = np.sort(q_reg)
    # q_reg_sort = q_reg_sort[::-1]
    # idx = np.argsort(q_reg)
    #
    # c = 0.9
    # for i in range(nx):
    #     if np.sum(q_reg_sort[0:i]) > c*np.sum(q_reg_sort):
    #         break
    #
    # src_selection = idx[nx-i-1:nx]
    # print(src_selection)
    # for i, j in itertools.product(src_selection, repeat=2):
    #     if i == j:
    #         continue
    #
    #     target = string_link(i, emod)
    #     src = string_link(j, emod)
    #
    #     link = target+'->'+src
    #     print(link)
    #     a_init[:] = a_f[:]
    #     a_init[:, j*emod: (j+1)*emod, i*emod: (i+1)*emod] = 0
    #     model_r = NeuraLVAR(p, use_lapack=False)
    #     model_r.fit(y.T, f, r, lambda_f, a_init=a_init, q_init=q_f*1, restriction=link, alpha=0.5, beta=0.1, **kwargs)
    #
    #     a_r = model_r._ravel_a(model_r._parameters[0])
    #     q_r = model_r._parameters[2]
    #     x_r = model_r._parameters[4]
    #     bias_r = model_r.compute_bias(y.T)
    #
    #     D_raw[j, i] = -2 * (model_r._lls[-1] - model_f._lls[-1])
    #     D[j, i] = -2 * (model_r._lls[-1] - model_f._lls[-1]) - (bias_f - bias_r)
    #
    #
    # alpha = 0.1
    # J = fdr_control(D, p, alpha)
    # plt.imshow(D[0:10, 0:10], vmin=0, vmax=150)
    # plt.colorbar()
    # plt.show()
    # print(D)

