import ipdb
import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg


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
    x_ = x_bar[:, :m]
    n = (x_bar.shape[0] - p + 1)

    # compute the following quantities carefully
    # s1 = x[2:].T.dot(x_bar[:-1]) / (x_bar.shape[0] - p + 1)
    s1 = x_[p:].T.dot(x_bar[p - 1:-1]) / n + s_cross.T

    # s2 = x_bar[:-1].T.dot(x_bar[:-1]) / (x_bar.shape[0] - p + 1)
    s2 = x_bar[p - 1:-1].T.dot(x_bar[p - 1:-1]) / n + s_bar

    # s3 = x[2:].T.dot(x[2:]) / (x_bar.shape[0] - p + 1)
    s3 = x_[p:].T.dot(x_[p:]) / n + s_bar[:m, :m]

    return s1, s2, s3


def solve_for_a(q, s1, s2, a, lambda2, max_iter=1000, tol=0.01, zeroed_index=None):
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
    q = np.diag(q)
    qinv = 1 / q
    # max step size:
    h_norm = np.sqrt((qinv ** 2).sum())
    h_norm *= np.sqrt((s2 ** 2).sum())
    tau = 0.99 / h_norm

    temp = np.empty_like(a)
    _a = np.empty_like(a)

    # gemm = linalg.get_blas_funcs("gemm", [a, s2])

    changes = [1.0]
    grad = np.zeros_like(a)
    for i in range(max_iter):
        if changes[-1] < tol:
            break
        _a[:] = a
        # Calculate gradient
        # grad = a.dot(s2) - s1
        # grad = a.dot(s2)
        grad = np.dot(a, s2, out=grad)
        grad -= s1
        # grad[:] = s1[:]
        # gemm(alpha=1.0, a, s2, beta=-1.0, c=grad, overwrite_c=True)
        grad *= qinv[:, None]

        # Forward step
        temp[:] = a
        temp -= tau * grad

        # Backward (proximal) step
        a = np.fabs(temp, out=a)
        a -= lambda2 * tau
        a = np.clip(a, a_min=0, a_max=None, out=a)
        a = np.copysign(a, temp, out=a)

        if zeroed_index is not None:
            a[zeroed_index] = 0.0

        changes.append(np.sqrt(np.sum((_a - a) ** 2) / np.sum(_a ** 2)))
    return a, changes


def solve_for_q(q, s1, s2, a, lambda2):
    """One-step sol to learn q, state-noise covariance matrix

    Parameters
    ----------
    q : ndarray of shape (n_sources, n_sources)
    s1 : ndarray of shape (n_sources, n_sources)
    s2 : nndarray of shape (n_sources, n_sources*order)
    a : ndarray of shape (n_sources, n_sources*order)
    lambda2 : float

    Returns
    -------
    q : ndarray of shape (n_sources, n_sources)
    """
    diag_indices = np.diag_indices_from(q)
    q_ = q[diag_indices]
    temp = np.einsum('ij,ji->i', a, s2.T)
    signa = np.sign(a)
    signa *= q_[:, None]
    temp2 = np.einsum('ij,ji->i', signa, a.T)
    temp2 *= lambda2
    # q_ = s1[diag_indices] + temp2 - temp
    if s1.ndim == 2:
        q_[:] = s1[diag_indices]
    else:
        q_[:] = s1
    q_ += temp2
    q_ -= temp
    # q_ /= t
    q[diag_indices] = q_
    return q


def compute_ll(y, x_, s_, s_hat, a, f, q, r, m, n, p):
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
    """
    x = x_[:, :m]
    t = (x_.shape[0] - p + 1)

    diff1 = x[p:] - x_[p - 1:-1].dot(a.T)
    i_m = np.eye(m)
    diag_indices = np.diag_indices_from(i_m)
    if q.ndim == 1:
        q = np.diag(q)
    c = linalg.cholesky(q, lower=True)
    c_inv = linalg.solve_triangular(c, i_m, lower=True, check_finite=False)
    val = t * np.log(c_inv[diag_indices]).sum()
    diff1 = diff1.dot(c_inv)
    val -= (diff1 * diff1).sum() / 2

    diff2 = y[p:] - x[p:].dot(f.T)
    i_n = np.eye(n)
    diag_indices = np.diag_indices_from(i_n)
    c = linalg.cholesky(r, lower=True)
    c_inv = linalg.solve_triangular(c, i_n, lower=True, check_finite=False)
    val += t * np.log(c_inv[diag_indices]).sum()
    diff2 = diff2.dot(c_inv)
    val -= (diff2 * diff2).sum() / 2

    diag_indices = np.diag_indices(m)
    c = linalg.cholesky(s_hat[(p - 1) * m:, (p - 1) * m:], lower=True)
    val += (t - p) * np.log(c[diag_indices]).sum()
    c = linalg.cholesky(s_, lower=True)
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
    s1 = x[2:].T.dot(x_bar[:-1]) / (x_bar.shape[0] - p + 1)
    s2 = x_bar[:-1].T.dot(x_bar[:-1]) / (x_bar.shape[0] - p + 1)
    s3 = x[2:].T.dot(x[2:]) / (x_bar.shape[0] - p + 1)

    a_ = np.zeros((m, m * p))
    a_[:] = 0 * np.reshape(np.swapaxes(a, 0, 1), (m, m * p))
    q_ = 0.1 * q.copy()
    for _ in range(20):
        a_, changes = solve_for_a(q_, s1, s2, a_, lambda2=0.1, max_iter=1000, tol=0.01)
        q_ = solve_for_q(q_, s3, s1, a_, lambda2=0.1)
    ipdb.set_trace()
