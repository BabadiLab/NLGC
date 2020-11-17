import mne
import os
import glob
from nlgc._nlgc import *
import ipdb
from codetiming import Timer
from nlgc._nlgc import _prepare_leadfield_reduction
from nlgc._nlgc import _gc_extraction
from nlgc._stat import fdr_control
import warnings
warnings.filterwarnings('ignore')

from csslaging import e
from csslaging import er_cov as cov

from matplotlib import pyplot as plt

# kwargs = {'raw': 'tsss-1-8-causal-ica-apply', 'src': 'ico-4', 'parc': 'aparc', 'epoch': 'all', 'session': 'Cocktail'}
# e.set(**kwargs)
# kwargs = {'src': 'ico-4'}
# e.set(**kwargs)

def data_generation(m, n_eigenmodes):

    n = 155
    src_per_patch = 20
    g_ = np.random.randn(n, m*src_per_patch)

    ex_g = np.zeros((n, m*n_eigenmodes))
    g = np.zeros((n, m))
    for i in range(m):
        u, s, _ = np.linalg.svd(g_[:, i*src_per_patch: (i+1)*src_per_patch])
        ex_g[:, i*n_eigenmodes: (i+1)*n_eigenmodes] = u[:, :n_eigenmodes]
        g[:, i] = g_[:, i*src_per_patch: (i+1)*src_per_patch].mean(axis=1)

    ex_g /= np.sqrt(np.sum(ex_g ** 2, axis=0))
    g /= np.linalg.norm(g, ord=2)

    # Simulate AR process
    ###############################################################################

    p, t = 1, 1000

    q = 0.1*np.eye(m)
    q[0, 0] = 1
    q[1, 1] = 1


    a = np.zeros(p * m * m, dtype=np.float64)
    a.shape = (p, m, m)

    a[0, 0, 0] = 0.9
    a[0, 1, 1] = 0.9

    a[0, 0, 1] = -0.5
    # a[0, 2, 2] = 0.8
    # a[0, 3, 3] = 0.8
    # a[0, 4, 4] = -0.8
    # a[0, 5, 5] = -0.8
    # a[0, 6, 6] = -0.8

    # a[1, 0, 4] = -0.4
    # a[0, 1, 5] = 0.4
    # a[1, 3, 1] = 0.4

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

    ## Data generation
    ###############################################################################

    y = x.dot(g.T)
    px = y.dot(y.T).trace()

    noise = np.random.standard_normal(y.shape)
    pn = noise.dot(noise.T).trace()
    multiplier = 1e2 * pn / px

    y += noise / np.sqrt(multiplier)
    r_cov = 1 / multiplier

    return ex_g, g, x, y, r_cov, p, JG


def debiased_dev(dev_raw, bias_f, bias_r):
    d = dev_raw.copy()
    bias_mat = bias_r - bias_f

    d[d < 0] = 0
    d[d > 0] += bias_mat[d > 0]
    np.fill_diagonal(d, 0)
    d[d < 0] = 0
    return d

def missed_false_detection(J, J_est):

    n, _ = J.shape
    Jr = J.copy()
    Jr = Jr.reshape((n**2, 1))

    Jr_est = J_est.copy()
    Jr_est = Jr_est.reshape((n**2, 1))
    Jr_est = np.absolute(Jr_est)
    Jr_est[Jr_est != 0] = 1

    missed_det = np.sum(Jr) - np.sum(Jr_est*Jr)
    false_det = np.sum((Jr-Jr_est) == -1)

    return missed_det, false_det

if __name__ == "__main__":

    total_trial = 1
    n_eigenmodes = 2
    lambda_range = np.asanyarray([10, 5, 2, 1.5, 1.25, 1, 0.75, 5e-1, 2e-1, 1e-1, 5e-2, 2e-2])
    max_iter = 500
    max_cyclic_iter = 3
    tol = 1e-5
    sparsity_factor = 0.0
    n_segments = 1
    m = 4

    msd_det = np.zeros((total_trial, 1))
    fls_det = np.zeros((total_trial, 1))

    for k in range(0, total_trial):
        ex_g, g, x, y, r_cov, p, JG = data_generation(m, n_eigenmodes)

        t, n = y.shape
        _, m = x.shape
        f = ex_g
        # ROIs = list(range(0, m))
        ROIs = []
        d = np.zeros((m, m))
        d_b = np.zeros((m, m))
        tt = t // n_segments
        for n_ in range(0, n_segments):
            dev_raw, bias_r, bias_f, model_f, conv_flag, dev_raw_, bias_f_, bias_r_ = \
                _gc_extraction(y[n_ * tt: (n_ + 1) * tt, :].T, f, r=r_cov*np.eye(n), p=p, p1=p,
                               n_eigenmodes=n_eigenmodes, ROIs=ROIs, lambda_range=lambda_range,
                               max_iter=max_iter, max_cyclic_iter=max_cyclic_iter,
                               tol=tol, sparsity_factor=sparsity_factor)

            d_b += dev_raw
            d += debiased_dev(dev_raw, bias_f, bias_r)

        d_b /= n_segments
        d /= n_segments
        J = fdr_control(d, p*n_eigenmodes, 0.001)

        msd_det[k], fls_det[k] = missed_false_detection(JG, J)
        # np.set_printoptions(precision=5)
        # print('deviance: ')
        # print(d_b)

        print('missed: ', msd_det[k]/np.sum(JG))
        print('false : ', fls_det[k]/m/(m-1))
        # print('---------------------')

    # print('# of actives:', m_active)
    # print('# of inactives:', m_inactive)
    # print('Total:', m)
    print('Summary: ')
    print('hit rate: ', 1 - msd_det.mean()/np.sum(JG), ' (', (1 - msd_det/np.sum(JG)).std(), ')')
    print('false alarm : ', fls_det.mean()/m/(m-1), ' (', fls_det.std()/m/(m-1), ')')
    print('---------------------------------')





