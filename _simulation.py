import mne
import os
import glob
from _nlgc import *
from _nlgc import _extract_label_eigenmodes as _extracted_fwd
import ipdb
from nlgc.core import gc_extraction
from matplotlib import pyplot as plt
import pickle
import logging


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

    idx0 = 0
    i0 = label_vertidxs[idx0][:]
    g0 = G[:, i0].dot(src_flips[idx0])

    # idx1 = 32
    # i1 = label_vertidxs[idx1][:]
    # g1 = G[:, i1].dot(src_flip[idx1])

    g0 = g0 / np.sqrt(g0.T.dot(g0))
    # g1 = g1 / np.sqrt(g1.T.dot(g1))

    # g = np.hstack((g0, g1))
    # g = g / (g.dot(g.T).trace())

    g = g0

    Y = X.dot(g.T)
    px = Y.dot(Y.T).trace()

    noise = np.random.standard_normal(Y.shape)
    pn = noise.dot(noise.T).trace()
    multiplier = pn/px


    Y = multiplier*Y + noise


    # whiten the data
    logger.info('Whitening data matrix.')
    # No need
    # save the data
    # pickle.dump(Y, 'sim_data.pickled')

    extracted_G, label_vertidx, src_flip = _extracted_fwd(forward, labels, gain.T, mode, n_eigenmodes, allow_empty=True)

    return Y, extracted_G


def simulate_AR_process():
    m, p, t = 1, 1, 300
    np.random.seed(0)

    q = 0.1 * np.eye(m)
    q[0, 0] = 10
    # q[1, 1] = 110

    a = np.zeros(p * m * m, dtype=np.float64)
    a.shape = (p, m, m)

    a[0, 0, 0] = -0.8

    # a[0, 1, 1] = 0.6

    # a[0, 0, 1] = 0.7

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


def simulation_test():
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


if __name__ == '__main__':
    filename = os.path.realpath(os.path.join(__file__, '..', '..', "debug.log"))
    logging.basicConfig(filename=filename, level=logging.DEBUG)

    behrad_root = r"G:\My Drive\behrad\Aging"

    evoked = mne.read_evokeds(os.path.join(behrad_root, 'test', 'R2533-pass_single_M02-ave.fif'))
    forward = mne.read_forward_solution(os.path.join(behrad_root, 'test', 'R2533-ico-4-fwd.fif'))
    er_cov = mne.read_cov(os.path.join(behrad_root, 'test', 'emptyroom-cov.fif'))
    fname_labels = os.path.join(behrad_root, 'test', 'labels', 'R2533-*.label')
    labels = [mne.read_label(fname_label) for fname_label in glob.glob(fname_labels)]

    print("generate data!")
    n_eigenmodes = 1
    p = 1
    y, f_ = simulate_data(evoked[0], forward, er_cov, labels, n_eigenmodes=n_eigenmodes)
    print(f_.shape)

    # f = f_[:, 150:155]
    f = f_[:, 0:5]

    print("behrad!")

    # y, f, p, n_eigenmodes = simulation_test()

    d_raw, bias_r, bias_f, a_f, q_f = gc_extraction(y, f, p=p, n_eigenmodes=n_eigenmodes, alpha=0.1, beta=0.2)

