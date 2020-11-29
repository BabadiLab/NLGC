# author: Behrad Soleimani <behrad@umd.edu>

from nlgc._nlgc import *
import ipdb
from nlgc._nlgc import _nlgc_map_opt
import matplotlib
import matplotlib.pyplot as plt
from table import missed_false_detection
import warnings
warnings.filterwarnings('ignore')


def data_generation():

    n = 15
    m = 3
    m_active = 2
    p, t = 2, 3000

    q = 0.01*np.eye(m)

    a = np.zeros(p * m * m, dtype=np.float64)
    a.shape = (p, m, m)

    for i in range(m_active):
        q[i, i] = 1
        a[1, i, i] = 0.9

    if m_active >= 2:
        a[0, 0, 1] = -0.5

    if m_active >= 3:
        a[0, 0, 2] = -0.5

    if m_active >= 4:
        a[0, 3, 2] = -0.5

    if m_active >= 5:
        a[0, 3, 4] = -0.5

    temp_JG = np.sum(np.abs(a), axis=0)
    JG = temp_JG != 0
    np.fill_diagonal(JG, 0)

    u = np.random.standard_normal(m * t)
    u.shape = (t, m)
    l = linalg.cholesky(q, lower=True)
    u = u.dot(l.T)

    x = np.empty((t, m), dtype=np.float64)
    for i in range(p):
        x[i] = 0.0

    for i in range(p, t):
        x[i] = u[i]
        for k in range(p):
            x[i] += a[k].dot(x[i - k - 1])

    f = np.random.randn(n, m)
    f /= np.sqrt(np.sum(f ** 2, axis=0))

    y = x.dot(f.T)
    px = y.dot(y.T).trace()

    noise = np.random.standard_normal(y.shape)
    pn = noise.dot(noise.T).trace()
    multiplier = 1e2 * pn / px

    y += noise / np.sqrt(multiplier)
    r_cov = 1 / multiplier

    return f, y, r_cov, p, JG

if __name__ == "__main__":

    np.random.seed(0)

    f, y, r_cov, p, JG = data_generation()

    n_eigenmodes = 1
    lambda_range = np.asanyarray([1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    max_iter = 500
    max_cyclic_iter = 3
    tol = 1e-5
    sparsity_factor = 0.0
    n_segments = 3
    var_thr = 0.95

    t, n = y.shape
    _, m = f.shape
    ROIs = list(range(0, m))

    temp_obj = _nlgc_map_opt('simulation', y.T, f, r=r_cov, p=p, p1=p, n_eigenmodes=n_eigenmodes, ROIs=ROIs,
                             lambda_range=lambda_range, n_segments=n_segments , var_thr=var_thr,
                             max_iter=max_iter, max_cyclic_iter=max_cyclic_iter, tol=tol,
                             sparsity_factor=sparsity_factor)

    J = temp_obj.fdr(alpha=0.1)

    msd_det, fls_det = missed_false_detection(JG, J)
    print('################################################################')
    print('missed: ', msd_det / np.sum(JG))
    print('false : ', fls_det / (m * (m - 1) - np.sum(JG)))
    print('################################################################')
    np.set_printoptions(precision=2)
