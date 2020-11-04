import ipdb
import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg
from numba import jit, njit, float32, float64, vectorize, uint

np.seterr(all='warn')
import warnings


def calculate_ss(x_bar, s_bar, b, m, p):
    """Calculates the required second order expectations

    Parameters
    ----------
    x_bar : ndarray of shape (n_samples, n_sources*order)
        smoothed means
    s_bar : ndarray of shape (n_sources*order, n_sources*order)
        smoothed covariances
    b : ndarray of shape (n_sources*order, n_sources*order)
        smoother gain
    m : int
        n_sources
    p : int
        order
    Returns
    -------
    s1 : ndarray of shape (n_sources, n_sources*order)
        n, n-1
    s2 : ndarray of shape (n_sources*order, n_sources*order)
        n-1, n-1 (augmented)
    s3 : ndarray of shape (n_sources, n_sources)
        n, n

    Notes
    -----
    the scaling by 1/n normalizes the Q function by time samples.
    """

    s_cross = b.dot(s_bar[:, :m])
    x_ = x_bar[:, :m]
    n = (x_bar.shape[0] - p)

    # compute the following quantities carefully
    # s1 = x[2:].T.dot(x_bar[:-1]) / (x_bar.shape[0] - p + 1)
    s1 = x_[p:].T.dot(x_bar[p - 1:-1]) / n + s_cross.T

    # s2 = x_bar[:-1].T.dot(x_bar[:-1]) / (x_bar.shape[0] - p + 1)
    s2 = x_bar[p - 1:-1].T.dot(x_bar[p - 1:-1]) / n + s_bar

    # s3 = x[2:].T.dot(x[2:]) / (x_bar.shape[0] - p + 1)
    s3 = x_[p:].T.dot(x_[p:]) / n + s_bar[(p - 1) * m:, (p - 1) * m:]
    # s3 = x_[p:].T.dot(x_[p:]) / n + s_bar[:m, :m]

    return s1, s2, s3, n


def solve_for_a(q, s1, s2, a, p1, lambda2, max_iter=5000, tol=1e-3, zeroed_index=None, update_only_target=False,
        n_eigenmodes=1):
    if not update_only_target or zeroed_index is None:
        return _solve_for_a(q, s1, s2, a, p1, lambda2, max_iter=max_iter, tol=tol, zeroed_index=zeroed_index,
                            n_eigenmodes=n_eigenmodes)
    else:
        target = np.unique(zeroed_index[0])
        s1_ = s1[target]
        a_ = a[target]
        q_ = q[target, target]
        if q_.ndim == 1:
            q_ = q_[:, None]
        target_ = np.asarray(zeroed_index[0]) - min(zeroed_index[0])
        source = np.asarray(zeroed_index[1])
        a_, changes = _solve_for_a(q_, s1_, s2, a_, p1, lambda2, max_iter=max_iter, tol=tol,
                                   zeroed_index=(target_, source), n_eigenmodes=n_eigenmodes)
        a[target] = a_
        return a, changes


# @njit(cache=True)
def _solve_for_a(q, s1, s2, a, p1, lambda2, max_iter=5000, tol=1e-3, zeroed_index=None, n_eigenmodes=1):
    """Gradient descent to learn a, state-transition matrix

    Parameters
    ----------
    q : ndarray of shape (n_sources, n_sources)
    s1 : ndarray of shape (n_sources, n_sources*order)
    s2 : ndarray of shape (n_sources*order, n_sources*order)
    a : ndarray of shape (n_sources, n_sources*order)
    lambda2 : float
    max_iter : int, default=1000
    tol : float, default=0.01
    zeroed_index : tuple of lists, like ([3, 3, 3], [1, 2, 3]).
        forces these indexes of a to 0.0.

    Notes
    -----
    To learn restricted model for i --> j, pass ([j] * p, list(range(i, m*p, m)))
    as zeroed index.

    Returns
    -------
    a : ndarray of shape (n_sources, n_sources*order)
    changes : list of floats
    """
    if lambda2 == 0:
        try:
            a = linalg.solve(s2, s1.T, assume_a='pos') # s1 * (s2 ** -1)
        except linalg.LinAlgError:
            a = linalg.solve(s2, s1.T, assume_a='sym') # s1 * (s2 ** -1)
        return a.T, None

    eps = np.finfo(s1.dtype).eps
    q = np.diag(q)
    qinv = 1 / q
    # qinv = np.ones_like(q)
    qinv = np.expand_dims(qinv, -1)
    q_inv_sqrt = np.sqrt(qinv)

    d = np.sqrt(np.diag(s2))
    # d = np.ones_like(d)
    s2 = s2 / d[:, None]
    s2 = s2 / d[None, :]
    s1 = s1 / d[None, :]
    a = a * d[None, :]

    a = a * q_inv_sqrt
    s1 = s1 * q_inv_sqrt

    # max step size:
    # h_norm = qinv.max()
    h_norm = np.linalg.eigvalsh(s2).max()
    tau_max = 0.99 / h_norm

    _a = np.empty_like(a)
    temp = np.empty_like(a)
    m = a.shape[0]
    p = a.shape[1] // m

    changes = np.zeros(max_iter+1)
    fs = np.zeros(max_iter+1)
    changes[0] = 1
    num = 1
    f_old = -2 * np.einsum('ij,ji->i', a.T, s1).sum() + np.einsum('ij,ji->i', a.T, a.dot(s2)).sum()
    fs[0] = f_old
    temp1 = a.dot(s2)
    # grad = np.zeros_like(a)
    for i in range(max_iter):
        if changes[i] < tol or num == 0:
            break
        _a[:] = a
        # Calculate gradient
        grad = temp1
        grad -= s1
        # grad *= qinv
        grad *= 2
        # ipdb.set_trace()
        # grad = _take_care(grad, n_eigenmodes)
        # ipdb.set_trace()
        # # #************* make the self history = 0 from lag p1***********
        # for k in range(p1, p):
        #     grad.flat[k * m::(p * m + 1)] = 0.0
        # # # *************************************************************
        # if zeroed_index is not None:
        #     grad[zeroed_index] = 0.0

        # Find opt step-size
        warnings.filterwarnings('error')
        try:
            # tau = 0.5 * (grad * grad).sum() / (np.diag(grad.dot(s2.dot(grad.T))) * qinv.ravel()).sum()
            temp2 = grad.dot(s2.T)
            # den = ((temp2 * grad).sum(axis=1) * qinv.ravel()).sum()
            den = ((temp2 * grad).sum(axis=1)).sum()
            num = (grad * grad).sum()
            tau = 0.5 * num / den
            tau = max(tau, tau_max)
        except Warning:
            raise RuntimeError(f'Q possibly contains negative value {q.min()}')
        warnings.filterwarnings('ignore')

        while True:
            # Forward step
            temp = _a.copy()
            temp -= tau * grad

            # Backward (proximal) step
            a = shrink(temp, lambda2 * tau)
            # a = project(a, 1.0)

            # #************* make the self history = 0 from lag p1***********
            for k in range(p1, p):
                a.flat[k * m::(p * m + 1)] = 0.0
            # # *************************************************************
            if zeroed_index is not None:
                a[zeroed_index] = 0.0

            "************* make the cross history between eigenmodes = 0 from lag p1***********"
            for l in range(0, m, n_eigenmodes):
                for u in range(n_eigenmodes):
                    for v in range(n_eigenmodes):
                        if v != u:
                            a[l+v, l+u::m] = 0
                # a[l, l + 1::m] = 0
                # a[l + 1, l::m] = 0
            "*********************************************************************"

            temp1 = a.dot(s2)
            # f_new = -2 * np.einsum('ij,ji->i', a.T, s1 * qinv).sum() + np.einsum('ij,ji->i', a.T, temp1 * qinv).sum()
            f_new = -2 * np.einsum('ij,ji->i', a.T, s1).sum() + np.einsum('ij,ji->i', a.T, temp1).sum()
            diff = (a - _a)
            f_new_upper = f_old + (grad * diff).sum() + (diff ** 2).sum() / (2 * tau)
            # print(f_new, f_new_upper)
            if f_new < f_new_upper or tau / tau_max < 1e-10:
                break
            else:
                tau /= 2

        num = np.sum(diff ** 2)
        den = np.sum(_a ** 2)
        f_old = f_new
        changes[i+1] = 1 if den == 0 else np.sqrt(num / den)

        fs[i+1] = f_old
    # ipdb.set_trace()
    # print(f"grad max: {grad.max()}, grad min: {grad.min()}, lambda:{lambda2}")
    # print(f"starting f {fs[0]}, closing f {f_old}")
    a = a / d[None, :]
    a = a / q_inv_sqrt
    return a, changes


@vectorize([float32(float32, float32),
            float64(float64, float64)], cache=True)
def shrink(x, t):
    if x > t:
        return x - t
    elif x < -t:
        return x + t
    else:
        return 0


# @njit([float32(float32, uint),float64(float64, uint)],cache=True)
def _take_care(a, n_eigenmodes):
    a_ = np.empty_like(a)
    # p = np.floor_divide(a.shape[1], a.shape[0])
    for i in range(a.shape[0]):
        for j in range(np.floor_divide(a.shape[0], n_eigenmodes)):
            if i not in range(j * n_eigenmodes, (j + 1) * n_eigenmodes):
                for l in range(0, a.shape[1], a.shape[0]):
                    a_[i, l+j*n_eigenmodes:l+(j+1)*n_eigenmodes] = a[i, l+j*n_eigenmodes:l+(j+1)*n_eigenmodes].sum()
            else:
                for l in range(0, a.shape[1], a.shape[0]):
                    a_[i, l+j*n_eigenmodes:l+(j+1)*n_eigenmodes] = 0.0
                    a_[i, l+i] = a[i, l+i]
    return a_


def find_best_lambda_cv(cvsplits, lambda2_range, q, x_bar, s_bar, b, m, p, a, max_iter=5000, tol=1e-3,
        zeroed_index=None):
    args = (q, x_bar, s_bar, b, m, p, a, max_iter, tol, zeroed_index)
    mse_path = np.empty((len(cvsplits), len(lambda2_range)))
    for i in range(len(cvsplits)):
        for j, cross_fit in enumerate(_find_cross_fit_cv(i, cvsplits, lambda2_range, *args)):
            mse_path[i, j] = cross_fit
    return mse_path


def _find_cross_fit_cv(i, cvsplits, lambda2s, q, x_bar, s_bar, b, m, p, a, max_iter=5000, tol=1e-3, zeroed_index=None):
    train, test = cvsplits[i]
    s1_train, s2_train, _, _ = calculate_ss(x_bar[train], s_bar, b, m, p)
    s1_test, s2_test, _, _ = calculate_ss(x_bar[test], s_bar, b, m, p)
    for lambda2 in lambda2s:
        a, _ = solve_for_a(q, s1_train, s2_train, a, lambda2, max_iter, tol, zeroed_index)
        cross_fit = np.sum(np.einsum('ij,ji->i', np.dot(a, s2_test) - 2 * s1_test, a.T) / np.diag(q))
        yield cross_fit


def solve_for_a_cv(q_upper, x_, s_, b, m, p, a, lambda2=None, max_iter=5000, tol=1e-3,
        zeroed_index=None, max_n_lambda2=5, cv=5):
    from sklearn.model_selection import TimeSeriesSplit
    import ipdb;
    if isinstance(cv, int):
        kf = TimeSeriesSplit(n_splits=2 * cv)
        cvsplits = [split for split in kf.split(x_)][-cv:]
    else:
        kf = cv
        cvsplits = [split for split in kf.split(x_)]
    s1, s2, _, _ = calculate_ss(x_, s_, b, m, p)
    lambda2_max = (s1 / np.diag(q_upper)[:, None]).max()
    lambda2_range = lambda2_max / 5 ** (np.arange(max_n_lambda2) + 1)
    cv_lambda2_range = None
    cv_mse_path = None
    # ipdb.set_trace()
    while True:
        mse_path = find_best_lambda_cv(cvsplits, lambda2_range, q_upper, x_, s_, b, m, p, a, max_iter,
                                                     tol, zeroed_index)
        if cv_mse_path is None:
            cv_mse_path = mse_path
            cv_lambda2_range = lambda2_range
        else:
            cv_mse_path = np.append(cv_mse_path, mse_path, axis=1)
            cv_lambda2_range = np.append(cv_lambda2_range, lambda2_range)
        best_lambda2 = cv_lambda2_range[cv_mse_path.mean(axis=0).argmin()]
        if best_lambda2 != lambda2_range[-1]:
            break
        else:
            lambda2_range = lambda2_range / 5 ** max_n_lambda2
    # fig, ax = plt.subplots()
    # ax.semilogx(cv_lambda2_range, cv_mse_path.mean(axis=0))
    a_upper, changes = solve_for_a(q_upper, s1, s2, a, best_lambda2, max_iter, tol, zeroed_index)
    return a_upper, best_lambda2


def solve_for_q(q, s1, s2, s3, a, lambda2, alpha=0, beta=0,):
    """One-step sol to learn q, state-noise covariance matrix

    Parameters
    ----------
    q : ndarray of shape (n_sources, n_sources)
    s1 : ndarray of shape (n_sources, n_sources)
    s2 : nndarray of shape (n_sources, n_sources*order)
    a : ndarray of shape (n_sources, n_sources*order)
    lambda2 : float
    alpha: float, default = 0.5
    beta : float, default = 1

    Returns
    -------
    q : ndarray of shape (n_sources, n_sources)

    Notes
    -----
    non-zero alpha, beta values imposes Inv-Gamma(alpha*n/2 - 1, beta*n) prior on q's.
    This equivalent to alpha*n - 2 additional observations that sum to beta*n.
    """
    diag_indices = np.diag_indices_from(q)
    q__ = q[diag_indices]
    temp = np.einsum('ij,ji->i', a, s2.T)
    # signa = np.sign(a)
    # signa *= np.expand_dims(q_, -1)
    # temp2 = np.einsum('ij,ji->i', signa, a.T)
    # temp2 *= lambda2
    temp3 = np.einsum('ij,ji->i', s2 - a.dot(s3), a.T)
    # q_ = s1[diag_indices] + temp2 - temp
    if s1.ndim == 2:
        q_ = s1[diag_indices]
    else:
        q_ = s1
    # q_ += (temp2 - temp)
    q_ -= (temp + temp3)
    # q_ /= t
    q_ += beta
    q_ /= (1 + alpha)
    q[diag_indices] = np.abs(q_)
    rel_change = ((q__ - q_) ** 2).sum() / (q__ ** 2).sum()
    # if q.min() < 0:
    #     import ipdb;
    #     ipdb.set_trace()
    # q = np.absolute(q)
    # q[q < 0] = 0
    return q, rel_change


def compute_cross_ll(x_, a, q, m, p):
    """Computes log-likelihood (See README)

    Parameters
    ----------
    y : ndarray of shape (n_samples, n_channels)
    x_ : ndarray of shape (n_samples, n_sources*order)
    s_ : ndarray of shape (n_sources*order, n_sources*order)
    s_hat : ndarray of shape (n_sources*order, n_sources*order)
    a : ndarray of shape (n_sources, n_sources*order)
    f : ndarray of shape (n_channels, n_sources)
    q : ndarray of shape (n_sources, n_sources)
    r : ndarray of shape (n_channels, n_channels)
    m : int
        n_sources
    n : int
        n_channels
    p : int
        order
    returns
    -------
    val : float
        the log-likelihood value
        note that it is not normalized by T.
    """
    x = x_[:, :m]
    t = (x_.shape[0] - p)

    diff1 = x[p:] - x_[p - 1:-1].dot(a.T)
    # i_m = np.eye(m)
    # diag_indices = np.diag_indices_from(i_m)
    # if q.ndim == 1:
    #     q = np.diag(q)
    # c = linalg.cholesky(q, lower=True)
    # c_inv = linalg.solve_triangular(c, i_m, lower=True, check_finite=False)
    # # val = t * np.log(c_inv[diag_indices]).sum()
    # diff1 = diff1.dot(c_inv.T)
    val = -(diff1 * diff1).sum() / 2

    return val


def compute_ll(y, x_, s, s_, s_hat, a, f, q, r, m, n, p):
    """Computes log-likelihood (See README)

    Parameters
    ----------
    y : ndarray of shape (n_samples, n_channels)
    x_ : ndarray of shape (n_samples, n_sources*order)
    s_ : ndarray of shape (n_sources*order, n_sources*order)
    s_hat : ndarray of shape (n_sources*order, n_sources*order)
    a : ndarray of shape (n_sources, n_sources*order)
    f : ndarray of shape (n_channels, n_sources)
    q : ndarray of shape (n_sources, n_sources)
    r : ndarray of shape (n_channels, n_channels)
    m : int
        n_sources
    n : int
        n_channels
    p : int
        order
    returns
    -------
    val : float
        the log-likelihood value
        note that it is not normalized by T.
    """
    x = x_[:, :m]
    t = (x_.shape[0] - p)

    diff1 = x[p:] - x_[p - 1:-1].dot(a.T)
    i_m = np.eye(m)
    diag_indices = np.diag_indices_from(i_m)
    if q.ndim == 1:
        q = np.diag(q)
    c = linalg.cholesky(q, lower=True)
    c_inv = linalg.solve_triangular(c, i_m, lower=True, check_finite=False)
    val = t * np.log(c_inv[diag_indices]).sum()
    diff1 = diff1.dot(c_inv.T)
    val -= (diff1 * diff1).sum() / 2

    diff2 = y[p:] - x[p:].dot(f.T)
    i_n = np.eye(n)
    diag_indices = np.diag_indices_from(i_n)
    c = linalg.cholesky(r, lower=True)
    c_inv = linalg.solve_triangular(c, i_n, lower=True, check_finite=False)
    val += t * np.log(c_inv[diag_indices]).sum()
    diff2 = diff2.dot(c_inv.T)
    val -= (diff2 * diff2).sum() / 2

    diag_indices = np.diag_indices(m)
    # c = linalg.cholesky(s_hat[(p - 1) * m:, (p - 1) * m:], lower=True)
    try:
        c = linalg.cholesky(s_hat[(p - 1) * m:, (p - 1) * m:], lower=True)
        val += t * np.log(c[diag_indices]).sum()
    except linalg.LinAlgError:
        c = linalg.eigvalsh(s_hat[(p - 1) * m:, (p - 1) * m:])
        val += t * np.log(c[c > 0]).sum()

    c = linalg.cholesky(s[:m, :m], lower=True)
    val += np.log(c[diag_indices]).sum()
    return val


def compute_Q(y, x_, s_, b, a, f, q, r, m, p):
    """Computes log-likelihood (See README)

    Parameters
    ----------
    y : ndarray of shape (n_samples, n_channels)
    x_ : ndarray of shape (n_samples, n_sources*order)
    s_ : ndarray of shape (n_sources*order, n_sources*order)
    s_hat : ndarray of shape (n_sources*order, n_sources*order)
    a : ndarray of shape (n_sources, n_sources*order)
    f : ndarray of shape (n_channels, n_sources)
    q : ndarray of shape (n_sources, n_sources)
    r : ndarray of shape (n_channels, n_channels)
    m : int
        n_sources
    n : int
        n_channels
    p : int
        order
    returns
    -------
    val : float
        the log-likelihood value
        note that it is not normalized by T.
    """
    x = x_[:, :m]

    # s1, s2, s3, t = calculate_ss(x_, s_, b, m, p)
    #
    # temp = s1.dot(a.T)
    # temp1 = s3 - temp - temp.T + a.dot(s2).dot(a.T)
    # if q.ndim == 2:
    #     q = np.diag(q)
    # qinv = np.diag(1 / q)
    # val = - t * (temp1 * qinv).sum() / 2

    # diff2 = y[p:] - x[p:].dot(f.T)
    # temp2 = diff2.T.dot(diff2) + t * f.dot(s_[:m, :m]).dot(f.T)
    # if r.ndim == 2:
    #     r = np.diag(r)
    # rinv = np.diag(1 / r)
    # val -= (temp2 * rinv).sum() / 2
    #
    # val -= t * np.log(q).sum() / 2
    #
    # val -= t * np.log(r).sum() / 2

    diff2 = y[p:] - x[p:].dot(f.T)
    val = -(diff2 * diff2).sum()

    return  val


def test_solve_for_a_and_q(t=1000):
    # n, m = 155, 6*2*68
    n, m, p, k = 3, 3, 2, 10
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
    # a = np.zeros(p * m * m, dtype=np.float64)
    # for i, val in zip(np.random.choice(p * m * m, k), np.random.randn(k)):
    #     a[i] = val
    # a.shape = (p, m, m)
    # a[0] /= 1.1 * linalg.norm(a[0])
    # a[1] /= 1.1 * linalg.norm(a[1])

    a = np.zeros(p * m * m, dtype=np.float64)
    a.shape = (p, m, m)

    a[1, 0, 1] = -0.2

    a[0, 0, 0] = 0.9

    a[0, 1, 1] = 0.9

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

    x_bar = np.empty((t - 1, m * p), dtype=np.float64)
    for _x, __x, x_ in zip(x[1:], x, x_bar):
        x_[:m] = _x
        x_[m:] = __x
    s1 = x[2:].T.dot(x_bar[:-1]) / (x_bar.shape[0] - p)
    s2 = x_bar[:-1].T.dot(x_bar[:-1]) / (x_bar.shape[0] - p)
    s3 = x[2:].T.dot(x[2:]) / (x_bar.shape[0] - p)

    a_ = np.zeros((m, m * p))
    a_[:] = 0 * np.reshape(np.swapaxes(a, 0, 1), (m, m * p))
    q_ = 1 * q.copy()
    for _ in range(100):
        a_, changes = solve_for_a(q_, s1, s2, a_, p, lambda2=0.1, max_iter=1000, tol=1e-8)
        q_ = solve_for_q(q_, s3, s1, s2, a_, lambda2=0.1)

    a__ = np.zeros((m, m * p))
    a__[:] = 0 * np.reshape(np.swapaxes(a, 0, 1), (m, m * p))
    a__[:] = a_
    q__ = 1 * q.copy()
    i = 0; j = 1
    for _ in range(100):
        a__, changes = solve_for_a(q__, s1, s2, a__, p, lambda2=0.1, max_iter=1000, tol=1e-8,
                                   zeroed_index=[(i, i), (j, j + 3)], update_only_target=True)
        q__ = solve_for_q(q__, s3, s1, s2, a__, lambda2=0.1)

    ipdb.set_trace()

    import itertools
    for i, j in itertools.product((0,1,2), repeat=2):
        if i == j:
            continue
        a__ = np.zeros((m, m * p))
        a__[:] = 0 * np.reshape(np.swapaxes(a, 0, 1), (m, m * p))
        q__ = 1 * q.copy()
        for _ in range(100):
            a__, changes = solve_for_a(q__, s1, s2, a__, p, lambda2=0.1, max_iter=1000, tol=1e-8,
                                       zeroed_index=[(i,i), (j, j+3)])
            q__ = solve_for_q(q__, s3, s1, s2, a__, lambda2=0.1)
        warnings.filterwarnings('ignore')
        ipdb.set_trace()
