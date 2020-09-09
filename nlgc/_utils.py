"""Deviance calculation
One comment here:
1. It will be easier to calculate the bias terms first and then update the deviance difference.
the workflow should be following:
1. raw deviance differences, raw_deviance. (store it)
2. calculate the bias terms for the full model, bias_f. (store it) [note that, we only need to compute m terms,
                                                                        but currently you are computing m*m times.]
    You can compute bias_f before computing the reduced models.
3. calculate the bias terms for the reduced models, bias_r (store it).

Once you finish all the matrices just do, raw_deviance + bias_f - bias_r or so!
That will ease debugging, streamline the coding flow.
"""


import numpy as np
from scipy import linalg
from math import log
import warnings
from .opt.m_step import calculate_ss
from .opt.e_step import sskf


def sample_path_bias(q, a, x_bar, zeroed_index):
    """Computes the bias in the deviance

    Parameters
    ----------
    q:  ndarray of shape (n_sources*mo, n_sources*mo)
    a:  ndarray of shape (n_sources*mo, n_sources*order*mo)
    x_bar:  ndarray of shape (t, n_sources*mo)
    idx_src: source index

    Returns
    -------
    bias

    """
    t, dxm = x_bar.shape
    _, dtot = a.shape
    p = dtot // dxm  # what's this??

    # ldot = np.zeros((dtot, 1))
    # ldotdot = np.zeros((dtot, dtot))
    bias = 0
    qd = np.diag(q)
    cx = np.zeros((t - p, dtot))

    for idx_src in range(0, dxm):
        ai = a[idx_src]  # in python slicing returns 1d array, so transpose is meaningless.

        xi = x_bar[p:, idx_src]   # p:t, -> p:, pythonic usage

        for k in range(p):
            cx[:, k*dxm:(k+1)*dxm] = x_bar[p-1-k:t-1-k]
            # for i in range(dxm):
                # cx[:, i*p + k] = x_bar[p - 1 - k: t - 1 - k, i]
                # cx[:, i*p + k] = x_bar[p - 1 - k: t - 1 - k, i]

        # gradient of log - likelihood

        # ldot[0, 0] = -t / qd[idx_src] / 2 + 1 / ((qd[idx_src] ** 2) / 2 * np.linalg.norm(xi - cx @ ai, ord=2))
        ldot = cx.T.dot(xi - cx.dot(ai)) / qd[idx_src]
        ldotdot = -cx.T.dot(cx) / qd[idx_src]

        if zeroed_index is not None:
            x_index, y_index = zeroed_index
            if idx_src in x_index:
                removed_idx = list(np.asanyarray(y_index)[np.asanyarray(x_index) == idx_src])
                ldot = np.delete(ldot, removed_idx)
                ldotdot = np.delete(ldotdot, removed_idx, axis=0)
                ldotdot = np.delete(ldotdot, removed_idx, axis=1)

                # ldot[list(np.asanyarray(y_index)[np.asanyarray(x_index) == idx_src])] = 0.0
        # hessian of log - likelihood

        # ldotdot[0, 0] = t / (qd[idx_src] ** 2) / 2 - 1 / ((qd[idx_src] ** 3) / np.linalg.norm(xi - cx @ ai, ord=2))
        # ldotdot[1:, 0] = -1 / (qd[idx_src] ** 2) * cx.T @ (xi - cx @ ai)
        # ldotdot[0, 1:] = ldotdot[1:, 0].T

        bias += ldot.dot(np.linalg.solve(ldotdot, ldot))
        # import ipdb; ipdb.set_trace()
    return bias


def mybias(idx_src, q, a, x_bar, s_bar, b, m, p):
    """Computes the bias in the deviance (proloy@umd.edu)

    Parameters
    ----------
    q:  ndarray of shape (n_sources*mo, n_sources*mo)
    a:  ndarray of shape (n_sources*mo, n_sources*order*mo)
    x_bar:  ndarray of shape (t, n_sources*mo)
    idx_src: source index

    Returns
    -------
    bias

    """
    _, dtot = a.shape

    n = (x_bar.shape[0] - p)

    #### These just uses means
    # x_ = x_bar[:, :m]
    # # compute the following quantities carefully
    # # s1 = x[2:].T.dot(x_bar[:-1]) / (x_bar.shape[0] - p + 1)
    # s1 = x_[p:].T.dot(x_bar[p - 1:-1]) / n
    #
    # # s2 = x_bar[:-1].T.dot(x_bar[:-1]) / (x_bar.shape[0] - p + 1)
    # s2 = x_bar[p - 1:-1].T.dot(x_bar[p - 1:-1]) / n
    #
    # # s3 = x[2:].T.dot(x[2:]) / (x_bar.shape[0] - p + 1)
    # s3 = x_[p:].T.dot(x_[p:]) / n

    ### These uses the whole distribution.
    s1, s2, s3 = calculate_ss(x_bar, s_bar, b, m, p)

    ai = a[idx_src]  # in python slicing returns 1d array, so transpose is meaningless.
    qi = q[idx_src, idx_src]


    ldot = np.empty((dtot + 1))
    ldotdot = np.empty((dtot + 1, dtot + 1))

    temp1 = s2.dot(ai)
    pev = s3[idx_src,idx_src] - 2 * s1[idx_src].dot(ai) +  ai.dot(temp1)  # prediction_error_variance
    temp1 -= s1[idx_src]

    # ldot[0, 0] = -t / qd[idx_src] / 2 + 1 / (qd[idx_src] ** 2) / 2 * np.linalg.norm(xi - cx @ ai, ord=2)
    ldot[0] = (- 1  + pev / qi) / (2 * qi)
    ldot[1:] = - temp1
    ldot[1:] /= qi

    # ldotdot[0, 0] = t / (qd[idx_src] ** 2) / 2 - 1 / (qd[idx_src] ** 3) / 2 * np.linalg.norm(xi - cx @ ai, ord=2)
    ldotdot[0, 0] = (0.5 - pev / qi)
    ldotdot[0, 1:] = temp1
    ldotdot[0] /= (qi * qi)
    ldotdot[1:, 1:] = - s2 / qi
    ldotdot[1:, 0] = ldotdot[0, 1:]

    # print(ldot)
    # print(ldotdot)

    try:
        c, low = linalg.cho_factor(-ldotdot)
        temp = linalg.cho_solve((c, low), ldot)
        bias = n * ldot.dot(temp)
    except linalg.LinAlgError:
        warnings.warn('source-index {:d} ldotdot is not negative definite: '
                      'setting positive eigenvalues equal to zero, '
                      'result may not be accurate.'.format(idx_src), RuntimeWarning, stacklevel=2)
        e, v = linalg.eigh(-ldotdot)
        temp = v.dot(ldot)
        idx = e > 0
        bias = np.sum(temp[idx] ** 2 / e[idx])
        bias *= n
        # temp = linalg.solve(-ldotdot, ldot, assume_a='sym')
    # import ipdb; ipdb.set_trace()
    # import ipdb;
    # ipdb.set_trace()
    return bias


def bias(q, a, x_bar, idx_src):
    """Computes the bias in the deviance

    Parameters
    ----------
    q:  ndarray of shape (n_sources*mo, n_sources*mo)
    a:  ndarray of shape (n_sources*mo, n_sources*order*mo)
    x_bar:  ndarray of shape (t, n_sources*mo)
    idx_src: source index

    Returns
    -------
    bias

    """
    t, dxm = x_bar.shape
    _, dtot = a.shape
    p = dtot // dxm  # what's this??

    ai = a[idx_src]  # in python slicing returns 1d array, so transpose is meaningless.

    xi = x_bar[p:, idx_src]   # p:t, -> p:, pythonic usage

    cx = np.zeros((t-p, dtot))

    # for i in range(dxm):
    #     for k in range(p):
    #         cx[:, i*p + k] = x_bar[p - 1 - k: t - 1 - k, i]
    for k in range(p):
        cx[:, k * dxm:(k + 1) * dxm] = x_bar[p - 1 - k:t - 1 - k]

    qd = np.diag(q)

    # gradient of log - likelihood
    ldot = np.zeros((dtot, 1))

    # ldot[0, 0] = -t / qd[idx_src] / 2 + 1 / ((qd[idx_src] ** 2) / 2 * np.linalg.norm(xi - cx @ ai, ord=2))
    ldot = 1 / qd[idx_src] * cx.T @ (xi - cx @ ai)

    # hessian of log - likelihood
    ldotdot = np.zeros((dtot, dtot))

    # ldotdot[0, 0] = t / (qd[idx_src] ** 2) / 2 - 1 / ((qd[idx_src] ** 3) / np.linalg.norm(xi - cx @ ai, ord=2))
    # ldotdot[1:, 0] = -1 / (qd[idx_src] ** 2) * cx.T @ (xi - cx @ ai)
    # ldotdot[0, 1:] = ldotdot[1:, 0].T
    ldotdot = -1 / qd[idx_src] * (cx.T @ cx)

    # import ipdb; ipdb.set_trace()
    return ldot.T@np.linalg.solve(ldotdot, ldot)


def test_bias():
    q = np.array([[2, 0], [0, 6]])
    a = np.array([[0.9, 0], [-0.5, 0.3]])
    x_ = np.array([[0.5, 0.5], [-0.5, -0.3], [0.2, 0.7], [-0.1, 0.4]])
    idx_src = 1
    dev_bias = bias(q, a, x_, idx_src)
    print(dev_bias)


def debiased_dev(xr_, xf_, ar, af, qr, qf, t, idx_reg, mo=1):
    """Computes debiased deviance

    Parameters
    ----------
    xr_: (reduced model) ndarray of shape (t, n_sources*order*mo)
    xf_: (full model) ndarray of shape (t, n_sources*order*mo)
    ar:  (reduced model) ndarray of shape (n_sources*mo, n_sources*order*mo)
    af:  (full model) ndarray of shape (n_sources*mo, n_sources*order*mo)
    qr:  (reduced model) ndarray of shape (n_sources*mo, n_sources*mo)
    qf:  (full model) ndarray of shape (n_sources*mo, n_sources*mo)
    t:   number of time samples
    idx_reg: region index
    mo:   number of eigen modes

    Returns
    -------
    deviance : debiased deviance

    """

    deviance = 0
    for idx_src in range(idx_reg * mo, (1 + idx_reg) * mo):
        deviance += t * (log(qr[idx_src, idx_src]) - log(qf[idx_src, idx_src]))
        deviance -= bias(qf, af, xf_, idx_src) - bias(qr, ar, xr_, idx_src)

    return deviance


def my_debiased_dev(fullmodel, reducedmodel, y, idx_reg, mo=1):
    """Computes debiased deviance (proloy@umd.edu)

    Parameters
    ----------
    xr_: (reduced model) ndarray of shape (t, n_sources*order*mo)
    xf_: (full model) ndarray of shape (t, n_sources*order*mo)
    ar:  (reduced model) ndarray of shape (n_sources*mo, n_sources*order*mo)
    af:  (full model) ndarray of shape (n_sources*mo, n_sources*order*mo)
    qr:  (reduced model) ndarray of shape (n_sources*mo, n_sources*mo)
    qf:  (full model) ndarray of shape (n_sources*mo, n_sources*mo)
    t:   number of time samples
    idx_reg: region index
    mo:   number of eigen modes

    Returns
    -------
    deviance : debiased deviance

    """
    t = y.shape[1]
    xs = None
    deviance = fullmodel.ll - reducedmodel.ll
    for idx_src in range(idx_reg * mo, (1 + idx_reg) * mo):
        emp_biases = []
        qs = []
        for model in (fullmodel, reducedmodel):
            y_, a_, a_upper, f_, q_, q_upper, _, r, xs, m, n, p, use_lapack = \
                model._prep_for_sskf(y, *model._parameters[:4])
            x_, s_, b, s_hat = sskf(y_, a_, f_, q_, r, xs=xs, use_lapack=use_lapack)
            emp_biases.append(mybias(idx_reg, q_upper, a_upper, x_, s_, b, m, p))
        deviance += emp_biases[0] - emp_biases[1]

    return deviance


def test_debiased_dev():
    n = 2
    mo = 2
    p = 1
    t = 10
    qf = np.diag(np.random.uniform(1, 10, n*mo))
    qr = qf + np.diag(np.random.uniform(0, 2, n*mo))

    af = np.random.uniform(0, 1, (n*mo, n*mo*p))
    ar = np.random.uniform(0, 1, (n*mo, n*mo*p))

    x_ = np.random.normal(0, 1, (t, n*mo))

    for idx_reg in range(n):
        dev_bias = debiased_dev(x_, x_, ar, af, qr, qf, t, idx_reg, mo)
        print(dev_bias)



if __name__ == '__main__':
    test_bias()