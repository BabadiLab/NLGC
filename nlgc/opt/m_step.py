import ipdb
import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg
from numba import jit, njit, float32, float64, vectorize


# (x_, s_, b)


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
    x_ = x_bar[:, (p - 1) * m:]
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


# @njit(cache=True)
def solve_for_a(q, s1, s2, a, lambda2, max_iter=5000, tol=1e-3, zeroed_index=None, beta=0.1):
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
    eps = np.finfo(s1.dtype).eps
    q = np.diag(q)
    qinv = 1 / q
    qinv = np.expand_dims(qinv, -1)
    # max step size:
    # h_norm = qinv.max()
    # h_norm *= np.linalg.eigvalsh(s2).max()
    # tau = 0.99 / h_norm

    _a = np.empty_like(a)
    m = a.shape[0]

    changes = np.zeros(max_iter+1)
    changes[0] = 1
    f_old = -2 * np.einsum('ij,ji->i', a.T, s1 * qinv).sum() + np.einsum('ij,ji->i', a.T, a.dot(s2) * qinv).sum()
    temp1 = a.dot(s2)
    # grad = np.zeros_like(a)
    for i in range(max_iter):
        if changes[i] < tol:
            break
        _a[:] = a
        # Calculate gradient
        grad = temp1
        grad -= s1
        grad *= qinv

        # import ipdb; ipdb.set_trace()
        if zeroed_index is not None:
            grad[zeroed_index] = 0.0

        # Find opt step-size
        # tau = 0.5 * (grad * grad).sum() / (np.diag(grad.dot(s2.dot(grad.T))) * qinv.ravel()).sum()
        temp = grad.dot(s2.T)
        den = ((temp * grad).sum(axis=1) * qinv.ravel()).sum()
        num = (grad * grad).sum()
        tau = 0.5 * num / den

        while True:
            # Forward step
            temp[:] = _a
            temp -= tau * grad

            # Backward (proximal) step
            a = shrink(temp, lambda2 * tau)

            #************* make the self history = 0***********
            # a.flat[::(2*m+1)] = 0.0
            # a.flat[m::(2*m+1)] = 0.0
            # ***************************************************

            if zeroed_index is not None:
                a[zeroed_index] = 0.0

            temp1 = a.dot(s2)
            f_new = -2 * np.einsum('ij,ji->i', a.T, s1 * qinv).sum() + np.einsum('ij,ji->i', a.T, temp1 * qinv).sum()
            diff = (a - _a)
            f_new_upper = f_old + (grad * diff).sum() + (diff ** 2).sum() / (2 * tau)
            # print(f_new, f_new_upper)
            if f_new < f_new_upper or tau < eps:
                break
            else:
                tau /= 2

        num = np.sum(diff ** 2)
        den = np.sum(_a ** 2)
        f_old = f_new
        changes[i+1] = 1 if den == 0 else np.sqrt(num / den)
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
    q_ = q[diag_indices]
    temp = np.einsum('ij,ji->i', a, s2.T)
    # signa = np.sign(a)
    # signa *= np.expand_dims(q_, -1)
    # temp2 = np.einsum('ij,ji->i', signa, a.T)
    # temp2 *= lambda2
    temp3 = np.einsum('ij,ji->i', s2 - a.dot(s3), a.T)
    # q_ = s1[diag_indices] + temp2 - temp
    if s1.ndim == 2:
        q_[:] = s1[diag_indices]
    else:
        q_[:] = s1
    # q_ += (temp2 - temp)
    q_ -= (temp + temp3)
    # q_ /= t
    q_ += beta
    q_ /= (1 + alpha)
    q[diag_indices] = q_
    return q


def compute_cross_ll(y, x_, s_, s_hat, a, f, q, r, m, n, p):
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


def test_solve_for_a_and_q(t=1000):
    # n, m = 155, 6*2*68
    n, m, p, k = 4, 3, 2, 10
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

    x_bar = np.empty((t - 1, m * p), dtype=np.float64)
    for _x, __x, x_ in zip(x[1:], x, x_bar):
        x_[:m] = _x
        x_[m:] = __x
    s1 = x[2:].T.dot(x_bar[:-1]) / (x_bar.shape[0] - p)
    s2 = x_bar[:-1].T.dot(x_bar[:-1]) / (x_bar.shape[0] - p)
    s3 = x[2:].T.dot(x[2:]) / (x_bar.shape[0] - p)

    a_ = np.zeros((m, m * p))
    a_[:] = 0 * np.reshape(np.swapaxes(a, 0, 1), (m, m * p))
    q_ = 0.1 * q.copy()
    for _ in range(20):
        a_, changes = solve_for_a(q_, s1, s2, a_, lambda2=0.01, max_iter=1000, tol=1e-8)
        q_ = solve_for_q(q_, s3, s1, s2, a_, lambda2=0.01)
        # ipdb.set_trace()
    ipdb.set_trace()
