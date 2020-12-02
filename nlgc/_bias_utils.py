# Author: Behrad Soleimani <behrad@umd.edu>
# Author: Proloy Das <proloy@umd.edu>
"Deviance calculation"

import numpy as np
import warnings
from scipy import linalg

from .opt.m_step import calculate_ss


def sample_path_bias(q, a, x_bar, zeroed_index, n_eigenmodes):
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
    p = dtot // dxm

    bias = 0
    qd = np.diag(q)
    cx = np.zeros((t - p, dtot))

    for idx_src in range(dxm):
        ai = a[idx_src]

        xi = x_bar[p:, idx_src]

        for k in range(p):
            cx[:, k * dxm:(k + 1) * dxm] = x_bar[p - 1 - k:t - 1 - k]

        # gradient of log - likelihood
        ldot = cx.T.dot(xi - cx.dot(ai)) / qd[idx_src]
        # hessian of log - likelihood
        ldotdot = -cx.T.dot(cx) / qd[idx_src]

        # if zeroed_index is not None:
        #     x_index, y_index = zeroed_index
        #     if idx_src in x_index:
        #         removed_idx = list(np.asanyarray(y_index)[np.asanyarray(x_index) == idx_src])
        #         ldot = np.delete(ldot, removed_idx)
        #         ldotdot = np.delete(ldotdot, removed_idx, axis=0)
        #         ldotdot = np.delete(ldotdot, removed_idx, axis=1)

        # FIX removing cross-talk components (that forced to be zero)
        for l in range(0, dxm, n_eigenmodes):
            for u in range(n_eigenmodes):
                for v in range(n_eigenmodes):
                    if v != u and idx_src == l + v:
                        removed_idx = list(range(l + u, dtot, dxm))
                        if zeroed_index is not None:
                            x_index, y_index = zeroed_index
                            if idx_src in x_index:
                                removed_idx.extend(list(np.asanyarray(y_index)[np.asanyarray(x_index) == idx_src]))
                        ldot = np.delete(ldot, removed_idx)
                        ldotdot = np.delete(ldotdot, removed_idx, axis=0)
                        ldotdot = np.delete(ldotdot, removed_idx, axis=1)

        bias += ldot.dot(np.linalg.solve(ldotdot, ldot))
    return bias


def bias_by_idx(idx_src, q, a, x_bar, s_bar, b, m, p, zeroed_index=None):
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
    warnings.filterwarnings('always')
    _, dtot = a.shape

    ### These uses the whole distribution.
    s1, s2, s3, n = calculate_ss(x_bar, s_bar, b, m, p)

    ai = a[idx_src]  # in python slicing returns 1d array, so transpose is meaningless.
    qi = q[idx_src, idx_src]

    ldot = np.empty((dtot))
    ldotdot = np.empty((dtot, dtot))

    temp1 = s2.dot(ai)
    temp1 -= s1[idx_src]

    ldot[:] = - temp1
    ldot[:] /= qi

    ldotdot[:, :] = - s2 / qi

    if zeroed_index is not None:
        x_index, y_index = zeroed_index
        if idx_src in x_index:
            removed_idx = list(np.asanyarray(y_index)[np.asanyarray(x_index) == idx_src])
            ldot = np.delete(ldot, removed_idx)
            ldotdot = np.delete(ldotdot, removed_idx, axis=0)
            ldotdot = np.delete(ldotdot, removed_idx, axis=1)

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

    return bias


def debias_deviances(dev_raw, bias_f, bias_r):
    d = dev_raw.copy()
    bias_mat = bias_r - bias_f
    d[bias_r != 0] += bias_mat[bias_r != 0]
    np.fill_diagonal(d, 0)
    d[d < 0] = 0
    return d
