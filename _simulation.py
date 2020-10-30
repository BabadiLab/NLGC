import mne
import os
import glob
from _nlgc import *
from _nlgc import _extract_label_eigenmodes as _extracted_fwd
import ipdb
from matplotlib import pyplot as plt
import pickle
import logging
from scipy.stats import chi2
from nlgc._stat import fdr_control
import itertools
from _nlgc import _gc_extraction, NLGC

import warnings
warnings.filterwarnings('ignore')

def _undo_source_weighting(G, source_weighting):
    G = G / source_weighting[None, :]
    return G


def _extract_label_eigenmodes(fwd, labels, mode='svd_flip', allow_empty=False):
    "Zero columns corresponds to empty labels"
    from mne.source_space import SourceSpaces
    from mne.utils import (logger, _check_option, _validate_type)
    from mne.source_estimate import (_prepare_label_extraction, _BaseSourceEstimate,
                                     _BaseVolSourceEstimate,_BaseVectorSourceEstimate, SourceEstimate,
                                     MixedSourceEstimate, VolSourceEstimate)
    src = fwd['src']
    _validate_type(src, SourceSpaces)
    _check_option('mode', mode, ['svd', 'svd_flip'] + ['auto'])

    if len(src) > 2:
        if src[0]['type'] != 'surf' or src[1]['type'] != 'surf':
            raise ValueError('The first 2 source spaces have to be surf type')
        if any(np.any(s['type'] != 'vol') for s in src[2:]):
            raise ValueError('source spaces have to be of vol type')

        n_aparc = len(labels)
        n_aseg = len(src[2:])
        n_labels = n_aparc + n_aseg
    else:
        n_labels = len(labels)

    # create a dummy stc
    kind = src.kind
    vertno = [s['vertno'] for s in src]
    nvert = np.array([len(v) for v in vertno])
    if kind == 'surface':
        stc = SourceEstimate(np.empty(nvert.sum()),vertno, 0.0, 0.0, 'dummy', )
    elif kind == 'mixed':
        stc = MixedSourceEstimate(np.empty(nvert.sum()), vertno, 0.0, 0.0, 'dummy', )
    else:
        stc = VolSourceEstimate(np.empty(nvert.sum()), vertno, 0.0, 0.0, 'dummy', )
    stcs = [stc]

    vertno  = None
    for si, stc in enumerate(stcs):
        if vertno is None:
            vertno = copy.deepcopy(stc.vertices)  # avoid keeping a ref
            nvert = np.array([len(v) for v in vertno])
            label_vertidx, src_flip = \
                _prepare_label_extraction(stc, labels, src, mode.replace('svd', 'mean'),
                                          allow_empty)
        if isinstance(stc, (_BaseVolSourceEstimate,
                            _BaseVectorSourceEstimate)):
            _check_option(
                'mode', mode, ('svd',),
                'when using a volume or mixed source space')
            mode = 'svd' if mode == 'auto' else mode
        else:
            mode = 'svd_flip' if mode == 'auto' else mode

        logger.info('Extracting time courses for %d labels (mode: %s)'
                    % (n_labels, mode))

    return label_vertidx, src_flip

def check_uncorrelated_columns(g, thr):
    out = 0
    temp = np.corrcoef(g.T)
    np.fill_diagonal(temp, 0)
    temp = np.absolute(temp)
    if np.sum(temp > thr):
        out = 1

    return out

def simulate_data(evoked, forward, noise_cov, labels, loose=0.0, depth=0.8, maxit=10000, tol=1e-6, pca=True,
        rank=None, mode='svd_flip', n_eigenmodes=1,):
    from mne.forward import is_fixed_orient
    from mne.minimum_norm.inverse import _check_reference
    from mne.utils import logger, verbose, warn
    from mne.inverse_sparse.mxne_inverse import _prepare_gain

    _check_reference(evoked)

    forward, gain, gain_info, whitener, source_weighting, mask = _prepare_gain(
        forward, evoked.info, noise_cov, pca, depth, loose, rank)

    ### gain is already whitened!

    if not is_fixed_orient(forward):
        raise ValueError(f"Cannot work with free orientation forward: {forward}")

    # G = _undo_source_weighting(gain, source_weighting)
    G = gain.copy()
    label_vertidxs, src_flips = _extract_label_eigenmodes(forward, labels)

    X = simulate_AR_process()
    n, _ = G.shape
    _, m_active = X.shape

    auditory_ROIs = []
    grant_ROIs = ['rostralmiddlefrontal', 'caudalmiddlefrontal', 'parsopercularis', 'parstriangularis', 'superiortemporal',
                  'middletemporal', 'transversetemporal']
    for i in range(0, len(labels)):
        for j in grant_ROIs:
            if j in labels[i].name:
            # print(i, labels[i].name)
                auditory_ROIs.append(i)
    # [1, 6, 10, 19, 23, 31, 38, 43, 47, 49, 50, 64, 65, 66]
    auditory_ROIs = [1, 6, 10, 19, 23, 31, 38, 43, 47, 49, 50, 64, 65, 66]
    auditory_ROIs = [2, 6]

    G_temp = np.empty((155,1))

    # selected_ROIs = [0, 1, 2, 3, 4]
    selected_ROIs = auditory_ROIs
    g = np.empty((n, m_active))
    for i in range(0, m_active):
        idx = selected_ROIs[i]
        i0 = label_vertidxs[idx]
        # W = np.random.uniform(0,1, len(i0))
        # W = np.random.standard_normal(len(i0))
        # W /= np.sum(W)
        # ws = src_flips[idx]
        # w_ = np.zeros((len(i0), 1))
        # w_[:, 0] = np.diag(ws * W)
        # w_[:, 0] = ws


        g0 = G[:, i0].dot(src_flips[idx])
        g0 = g0 / len(i0)
        g0 = g0 / np.sqrt(g0.T.dot(g0))
        g[:, i] = g0[:, 0]

        ############################################
        g_ = np.zeros((n, len(i0)))
        for k in range(0, len(i0)):
            # g_[:, k] = src_flips[idx][k]*G[:, i0[k]]
            g_[:, k] = G[:, i0[k]]

        u, s, _ = np.linalg.svd(g_)
        k1 = u[0]#*np.sqrt(s[0])
        k2 = u[1]#*np.sqrt(s[1])
        k3 = u[2]#*np.sqrt(s[1])
        k4 = u[3]#*np.sqrt(s[1])
        k5 = u[4]#*np.sqrt(s[1])
        K = np.vstack((k1, k2, k3, k4, k5))
        G_temp = np.hstack((G_temp, K.T))


    extracted_G_ = G_temp[:, 1:]
    ############################################

    selected_ROIs = auditory_ROIs
    selected_ROIs_ = list()
    for k in range(0, len(selected_ROIs)):
        selected_ROIs_.append([i for i, value in enumerate(auditory_ROIs) if value == selected_ROIs[k]][0])

    # print(selected_ROIs)
    #
    Y = X.dot(g.T)
    px = Y.dot(Y.T).trace()

    noise = np.random.standard_normal(Y.shape)
    pn = noise.dot(noise.T).trace()
    multiplier = 1e2*pn/px

    Y += noise/np.sqrt(multiplier)
    r_cov = 1/multiplier

    # whiten the data
    logger.info('Whitening data matrix.')
    # No need
    # save the data
    # pickle.dump(Y, 'sim_data.pickled')

    extracted_G, _, _ = _extracted_fwd(forward, labels, gain.T, mode, n_eigenmodes, allow_empty=True)
    # extracted_G = extracted_G / np.sqrt(np.sum(extracted_G**2, axis=0))
    # extracted_G_ = np.empty((n, len(auditory_ROIs)*n_eigenmodes))
    # for i in range(0, len(auditory_ROIs)):
    #     extracted_G_[:, i*n_eigenmodes: (i+1)*n_eigenmodes] = extracted_G[:, auditory_ROIs[i]*n_eigenmodes: (auditory_ROIs[i]+1)*n_eigenmodes]

    ipdb.set_trace()

    # return extracted_G
    return Y, extracted_G_, selected_ROIs_, r_cov, X
    # return Y, extracted_G, selected_ROIs_, r_cov, X

def simulate_data_(evoked, forward, noise_cov, labels, loose=0.0, depth=0.8, maxit=10000, tol=1e-6, pca=True,
        rank=None, mode='svd_flip', n_eigenmodes=1,m=15):
    from mne.forward import is_fixed_orient
    from mne.minimum_norm.inverse import _check_reference
    from mne.utils import logger, verbose, warn
    from mne.inverse_sparse.mxne_inverse import _prepare_gain

    _check_reference(evoked)

    forward, gain, gain_info, whitener, source_weighting, mask = _prepare_gain(
        forward, evoked.info, noise_cov, pca, depth, loose, rank)

    ### gain is already whitened!

    if not is_fixed_orient(forward):
        raise ValueError(f"Cannot work with free orientation forward: {forward}")

    # G = _undo_source_weighting(gain, source_weighting)
    G = gain.copy()
    label_vertidxs, src_flips = _extract_label_eigenmodes(forward, labels)

    n, _ = G.shape

    import random
    # ROIs = random.sample(range(68), m)
    ROIs = list(range(0,m))
    f = np.zeros((n, len(ROIs)))
    c = 0
    for i in ROIs:
        i0 = label_vertidxs[i]
        # idx = random.sample(range(len(i0)), 1)
        idx = random.sample(range(3), 1)
        f0 = G[:, i0[idx]]*src_flips[i][idx]
        f0 = f0 / np.sqrt(f0.T.dot(f0))
        f[:, c] = f0[:, 0]
        c+=1

    y, f, p, r_cov, JG = sim_rnd_real_F(f)

    return y, f, p, r_cov, JG

def simulate_AR_process():
    m, p, t = 2, 1, 1000
    # np.random.seed(1)

    q =  1e-3*np.eye(m)
    # q[0, 0] = 1e-2
    # q[1, 1] = 1e-2
    # q[2, 2] = 1e-2
    # q[3, 3] = 1e-3

    a = np.zeros(p * m * m, dtype=np.float64)
    a.shape = (p, m, m)

    a[0, 0, 0] = 0.9
    a[0, 1, 1] = 0.9
    # a[0, 2, 2] = 0.5
    # a[0, 3, 3] = 0.5
    # a[0, 4, 4] = 0.5

    a[0, 1, 0] = -0.2
    # a[0, 2, 1] = -0.2


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

    return x

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

def simulation_rnd(f=None):
    #                       155, 136,   4
    n, nx, n_eigenmodes, p = 20, 15, 1, 4
    # np.random.seed(0)

    m = nx * n_eigenmodes

    t = 10000

    r = np.eye(n)
    q = np.eye(m)
    # q[0, 0] = 1
    # q[1, 1] = 1
    # q[2, 2] = 1
    # q[3, 3] = 1
    # q[4, 4] = 1
    # q[5, 5] = 1
    # q[6, 6] = 1


    a = np.zeros(p * m * m, dtype=np.float64)
    a.shape = (p, m, m)

    a[0, 0, 1] = -0.1
    a[1, 0, 7] = -0.3

    a[0, 1, 0] = -0.2
    a[0, 1, 3] = 0.3
    a[0, 1, 4] = -0.2

    a[1, 2, 7] = -0.35

    a[1, 4, 6] = 0.3

    a[1, 5, 4] = -0.3
    a[2, 5, 6] = 0.3

    a[2, 8, 9] = 0.2
    a[2, 9, 8] = -0.2

    a[3, 10, 12] = -0.25
    a[3, 11, 10] = 0.25
    a[3, 12, 11] = -0.25

    a[1, 13, 14] = -0.1
    a[1, 14, 13] = 0.15

    temp_JG = np.sum(np.abs(a), axis=0)
    JG = temp_JG != 0

    a[0, 0, 0] = 0.8
    a[1, 1, 1] = 0.9
    a[1, 5, 5] = 0.45
    a[1, 6, 6] = -0.45
    a[2, 8, 8] = -0.6
    a[2, 9, 9] = 0.5
    a[2, 13, 13] = 0.65
    a[2, 14, 14] = 0.65


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

    # for i in range(1, m):
    #     a[2, i, i] = 0.1

    sn = np.random.standard_normal((m + n) * t)
    u = sn[:m * t]
    u.shape = (t, m)
    l = linalg.cholesky(q, lower=True)
    u = u.dot(l.T)
    v = sn[m * t:]
    v.shape = (t, n)
    l = linalg.cholesky(r, lower=True)
    v = v.dot(l.T)

    if f is None:
        f = np.random.randn(n, m)
        # f = np.eye(n)
        # f /= np.sqrt(f ** 2).sum(axis=0)
        f_normalizing_factor = np.sqrt(np.sum(f ** 2, axis=0))
        f /= f_normalizing_factor[None, :]

    x = np.empty((t, m), dtype=np.float64)
    for i in range(p):
        x[i] = 0.0

    for i in range(p, t):
        x[i] = u[i]
        for k in range(p):
            x[i] += a[k].dot(x[i - k - 1])
    r_cov = 1
    y_ = x.dot(f.T)
    y = 10 * np.sqrt((v ** 2).sum() / (y_ ** 2).sum()) * y_ + v

    return y, f, p, n_eigenmodes, r_cov, JG

def sim_rnd_real_F(f):

    t, p = 1000, 4

    n, m = f.shape

    r = np.eye(n)
    q = 1e-3*np.eye(m)
    for i in range(0,15):
        q[i, i] = 1

    a = np.zeros(p * m * m, dtype=np.float64)
    a.shape = (p, m, m)

    a[0, 0, 1] = -0.1
    a[1, 0, 7] = -0.3

    a[0, 1, 0] = -0.2
    a[0, 1, 3] = 0.3
    a[0, 1, 4] = -0.2

    a[1, 2, 7] = -0.35

    a[1, 4, 6] = 0.3

    a[1, 5, 4] = -0.3
    a[2, 5, 6] = 0.3

    a[2, 8, 9] = 0.2
    a[2, 9, 8] = -0.2

    a[3, 10, 12] = -0.25
    a[3, 11, 10] = 0.25
    a[3, 12, 11] = -0.25

    a[1, 13, 14] = -0.1
    a[1, 14, 13] = 0.15

    temp_JG = np.sum(np.abs(a), axis=0)
    JG = temp_JG != 0

    a[0, 0, 0] = 0.8
    a[1, 1, 1] = 0.9
    a[1, 5, 5] = 0.45
    a[1, 6, 6] = -0.45
    a[2, 8, 8] = -0.6
    a[2, 9, 9] = 0.5
    a[2, 13, 13] = 0.65
    a[2, 14, 14] = 0.65

    # f_normalizing_factor = np.sqrt(np.sum(f ** 2, axis=0))
    # f /= f_normalizing_factor[None, :]

    sn = np.random.standard_normal((m + n) * t)
    u = sn[:m * t]
    u.shape = (t, m)
    l = linalg.cholesky(q, lower=True)
    u = u.dot(l.T)
    v = sn[m * t:]
    v.shape = (t, n)
    l = linalg.cholesky(r, lower=True)
    v = v.dot(l.T)

    x = np.empty((t, m), dtype=np.float64)
    for i in range(p):
        x[i] = 0.0

    for i in range(p, t):
        x[i] = u[i]
        for k in range(p):
            x[i] += a[k].dot(x[i - k - 1])

    r_cov = 1
    y_ = x.dot(f.T)
    y = 10 * np.sqrt((v ** 2).sum() / (y_ ** 2).sum()) * y_ + v

    return y, f, p, r_cov, JG



if __name__ == '__main__':


    filename = os.path.realpath(os.path.join(__file__, '..', '..', "debug.log"))
    logging.basicConfig(filename=filename, level=logging.DEBUG)

    # behrad_root = r"G:\My Drive\behrad\Aging"
    behrad_root = "/Users/behrad/Google Drive/behrad/Aging"
    evoked = mne.read_evokeds(os.path.join(behrad_root, 'test', 'R2533-pass_single_M02-ave.fif'))
    forward = mne.read_forward_solution(os.path.join(behrad_root, 'test', 'R2533-ico-4-fwd.fif'))
    er_cov = mne.read_cov(os.path.join(behrad_root, 'test', 'emptyroom-cov.fif'))
    fname_labels = os.path.join(behrad_root, 'test', 'labels', 'R2533-*.label')
    labels = [mne.read_label(fname_label) for fname_label in glob.glob(fname_labels)]

    n_eigenmodes = 1
    p = 1
    n_segments = 1

    alpha = 0
    beta = 0
    # lambda_range = np.asanyarray([5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01])
    lambda_range = None
    max_iter = 100
    max_cyclic_iter = 3
    tol = 1e-5
    sparsity_factor = 0.0

    total_trials = 5
    m = 40
    d = np.zeros((total_trials,m,m))
    d_ = np.empty_like(d)
    missed_det = np.zeros((total_trials,1))
    false_det = np.zeros((total_trials,1))

    for i in range(0, total_trials):
        y, f_, p, r_cov, JG = simulate_data_(evoked[0], forward, er_cov, labels, n_eigenmodes=n_eigenmodes, m=m)

        n, m = f_.shape

        dev_raw, bias_r, bias_f, model_f, conv_flag, dev_raw_, bias_f_, bias_r_ = _gc_extraction(y.T, f_, r=r_cov*np.eye(n), p=p, p1=p,
                       n_eigenmodes=n_eigenmodes, ROIs=list(range(0,m // n_eigenmodes)), alpha=alpha, beta=beta,
                       lambda_range=lambda_range, max_iter=max_iter, max_cyclic_iter=max_cyclic_iter,
                       tol=tol, sparsity_factor=sparsity_factor)
        d[i] = dev_raw.copy()
        print(np.floor(d[i]))
        J = fdr_control(d[i], p, 0.1)


        missed_det[i], false_det[i] = missed_false_detection(JG, J)
        print('missed: ', missed_det[i])
        print('false: ', false_det[i])


    # d_avg = d.mean(axis=0)
    # d_std = d.std(axis=0)
    # a_f = model_f._parameters[0]
    # q_f = model_f._parameters[2]


    # print(np.floor(d_avg))
    #
    # J = fdr_control(d_avg, p, 0.1)
    # fig , ax = plt.subplots(3)
    # ax[0].matshow(JG)
    # ax[1].matshow(J)
    # cax = ax[2].matshow(d_avg)
    # fig.colorbar(cax)


    # print(np.abs(a_f[0, ::2, ::2]) + np.abs(a_f[0, 1::2, 1::2]) )
    # with open('y_' + '.obj', 'wb') as fh:
    #     pickle.dump(y, fh)
    #
    # with open('f_' + '.obj', 'wb') as fh_:
    #         pickle.dump(f, fh_)



    # n_eigenmodes = 2
    # p = 1
    # n_segment = 3
    # total_trial = 20
    #
    # # d_trial = np.zeros((total_trial, 5, 5))
    # alpha_vec = [0.01, 0.05, 0.1, 0.2]
    #
    # missed_det = np.zeros((total_trial, len(alpha_vec)))
    # false_det = np.zeros((total_trial, len(alpha_vec)))
    #
    # for trial in range(0, total_trial):
    #     y, f, selected_ROIs = simulate_data(evoked[0], forward, er_cov, labels, n_eigenmodes=n_eigenmodes)
    #     tt, ny = y.shape
    #     t = tt // n_segment
    #     _, nnx = f.shape
    #     nx = nnx // n_eigenmodes
    #
    #     J_GT = np.zeros((nx, nx))
    #
    #     J_GT[selected_ROIs[1], selected_ROIs[2]] = 1
    #     J_GT[selected_ROIs[1], selected_ROIs[0]] = 1
    #     J_GT[selected_ROIs[0], selected_ROIs[1]] = 1
    #     # J_GT[selected_ROIs[3], selected_ROIs[4]] = 1
    #     # J_GT[selected_ROIs[4], selected_ROIs[3]] = 1
    #     # J_GT[selected_ROIs[3], selected_ROIs[2]] = 1
    #     # J_GT[selected_ROIs[3], selected_ROIs[0]] = 1
    #     # J_GT[selected_ROIs[4], selected_ROIs[2]] = 1
    #     # J_GT[selected_ROIs[1], selected_ROIs[4]] = 1
    #     # J_GT[selected_ROIs[3], selected_ROIs[1]] = 1
    #     d_avg = np.zeros((nx, nx))
    #
    #     print('trial: ', trial)
    #
    #     for n in range(0, n_segment):
    #         print('segment: ', n)
    #
    #         d_raw, bias_r, bias_f, a_f, q_f, lambda_f, _ = gc_extraction(y[n*t: (n+1)*t, :], f, p, n_eigenmodes, ROIs=selected_ROIs)
    #
    #         test_obj = NLGC('test', nx, ny, t, p, n_eigenmodes, d_raw, bias_f, bias_r, a_f, q_f, lambda_f)
    #
    #         d = test_obj.compute_debiased_dev()
    #         d_avg += d
    #
    #     d_avg /= n_segment
    #     print(d_avg[selected_ROIs, :][:, selected_ROIs])
    #     # d_trial[trial, :, :] = d_avg[selected_ROIs, :][:, selected_ROIs]
    #     # ipdb.set_trace()
    #
    #
    #
    #     for j in range(0, len(alpha_vec)):
    #         i = len(selected_ROIs)*(len(selected_ROIs)-1)
    #         alpha_bar = alpha_vec[j]*(i+1)/(2*i*np.log(i))
    #         threshold = np.ceil(chi2.isf(alpha_bar, p*n_eigenmodes))
    #         # J = fdr_control(d_avg[selected_ROIs, :][:, selected_ROIs], p*n_eigenmodes, alpha=alpha_vec[j])
    #         J = d_avg > threshold
    #         missed_det[trial, j], false_det[trial, j] = missed_false_detection(J_GT[selected_ROIs, :][:, selected_ROIs], J[selected_ROIs, :][:, selected_ROIs])
    #
    #     print('missed: ', missed_det[trial, :])
    #     print('false: ', false_det[trial, :])
    #
    #
    # missed_mean = np.mean(missed_det, axis=0)
    # missed_std = np.std(missed_det, axis=0)
    #
    # false_mean = np.mean(false_det, axis=0)
    # false_std = np.std(false_det, axis=0)
    #
    # x_axis = ['0.01', '0.05', '0.1', '0.2']
    # x_pos = np.arange(len(x_axis))
    #
    # fig, (ax_f, ax_m) = plt.subplots(2)
    # ax_m.bar(x_pos, missed_mean, yerr=missed_std, align='center', alpha=0.5, ecolor='black', capsize=10)
    # ax_f.bar(x_pos, false_mean, yerr=false_std, align='center', alpha=0.5, ecolor='black', capsize=10)
    #
    # ax_m.set_ylabel('Missed detections')
    # ax_f.set_ylabel('Flase detections')
    #
    # ax_m.set_xticks(x_pos)
    # ax_m.set_xticklabels(x_axis)
    # ax_f.set_xticks(x_pos)
    # ax_f.set_xticklabels(x_axis)
    # ax_m.set_xlabel('FDR rate (alpha)')
    # ax_m.yaxis.grid(True)
    # ax_f.yaxis.grid(True)
    #
    # plt.tight_layout()
    # plt.savefig('segments='+str(n_segment)+', trials='+str(total_trial)+', eigenmodes='+str(n_eigenmodes)+', p='+str(p)+', actives='+str(len(selected_ROIs))+', ROIs='+str(nx)+'.png')
    # plt.show()

    ## histogram of deviances


    # n_eigenmodes = 2
    # p = 2
    # n_segment = 5
    # total_trial = 10
    # # GC_d = np.zeros((1, total_trial*10))
    # # non_GC_d = np.zeros((1, total_trial*10))
    # GC_d = np.zeros((1, total_trial))
    # non_GC_d = np.zeros((1, total_trial))
    #
    # for trial in range(0, total_trial):
    #     y, f, selected_ROIs = simulate_data(evoked[0], forward, er_cov, labels, n_eigenmodes=n_eigenmodes)
    #     tt, ny = y.shape
    #     t = tt // n_segment
    #     _, nnx = f.shape
    #     nx = nnx // n_eigenmodes
    #
    #     d_avg = np.zeros((nx, nx))
    #
    #     print('trial: ', trial)
    #
    #     for n in range(0, n_segment):
    #         print('segment: ', n)
    #
    #         d_raw, bias_r, bias_f, a_f, q_f, lambda_f, _ = gc_extraction(y[n*t: (n+1)*t, :], f, p, n_eigenmodes, ROIs=selected_ROIs)
    #
    #         test_obj = NLGC('test', nx, ny, t, p, n_eigenmodes, d_raw, bias_f, bias_r, a_f, q_f, lambda_f)
    #
    #         d_avg += test_obj.compute_debiased_dev()
    #
    #     d_avg /= n_segment
    #     d_temp = d_avg[selected_ROIs, :][:, selected_ROIs]
    #     d_temp = np.reshape(d_temp, (1, len(selected_ROIs)**2))
    #     print(d_temp)
    #
    #     # GC_d[0, trial*10: trial*10 + 10] = d_temp[0, [1, 5, 7, 9, 15, 16, 17, 19, 22, 23]]
    #     # non_GC_d[0, trial*10: trial*10 + 10] = d_temp[0, [2, 3, 4, 8, 10, 11, 13, 14, 20, 21]]
    #
    #     GC_d[0, trial] = d_temp[0, 7]
    #     print('GC: ', d_temp[0, 7])
    #     non_GC_d[0, trial] = d_temp[0, 20]
    #     print('non_GC: ', d_temp[0, 20])
    #
    #
    # fig, ax = plt.subplots()
    #
    # ax.hist(GC_d[GC_d > 0], bins=10, label='GC')
    # ax.hist(non_GC_d[non_GC_d > 0], bins=5, label='non_GC')
    # ax.legend()
    # # plt.savefig('histogram_test.svg')
    # plt.show()
    #
    # # print(d_avg[selected_ROIs, :][:, selected_ROIs])
    # # d_trial[trial, :, :] = d_avg[selected_ROIs, :][:, selected_ROIs]
    # # ipdb.set_trace()

