# author: proloy Das <proloy@umd.edu>
# author: Behrad Soleimani <behrad@umd.edu>

import ipdb
import itertools
from nlgc.opt.opt import NeuraLVAR, NeuraLVARCV
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy import linalg, sparse
import copy
from mne.forward import is_fixed_orient
from mne.minimum_norm.inverse import _check_reference
from mne.utils import logger, verbose, warn
from mne.inverse_sparse.mxne_inverse import _prepare_gain
import ipdb
from nlgc._stat import fdr_control
import pickle
# from nlgc.core import gc_extraction, NLGC

def truncatedsvd(a, n_components=2):
    if n_components > min(*a.shape):
        raise ValueError('n_components={:d} should be smaller than '
                         'min({:d}, {:d})'.format(n_components, *a.shape))
    u, s, vh = linalg.svd(a, full_matrices=False, compute_uv=True,
                          overwrite_a=True, check_finite=True,
                          lapack_driver='gesdd')
    return vh[:n_components] * s[:n_components][:, None]


_svd_funcs = {
    'svd_flip': lambda flip, data, n_components: truncatedsvd(flip * data, n_components),
    'svd': lambda flip, data, n_components: truncatedsvd(data, n_components)
}


# Note for covariance, source_weighting needs to be applied twice!
def _reapply_source_weighting(X, source_weighting):
    X *= source_weighting[:, None]
    return X


def nlgc_map(name, evoked, forward, noise_cov, labels, p, n_eigenmodes=2, alpha=0, beta=0, ROIs_names=['just_full_model'], n_segments=1,
             loose=0.0, depth=0.8, pca=True, rank=None, mode='svd_flip',
             lambda_range=[1, 0.1, 0.01, 0.001], max_iter=50, max_cyclic_iter=5, tol=1e-8, sparsity_factor=0.1):
    _check_reference(evoked)

    forward, gain, gain_info, whitener, source_weighting, mask = _prepare_gain(
        forward, evoked.info, noise_cov, pca, depth, loose, rank)

    if not is_fixed_orient(forward):
        raise ValueError(f"Cannot work with free orientation forward: {forward}")

    # get the data
    sel = [evoked.ch_names.index(name) for name in gain_info['ch_names']]
    M = evoked.data[sel]

    # whiten the data
    logger.info('Whitening data matrix.')
    M = np.dot(whitener, M)
    M /= np.linalg.norm(M, ord='fro')
    ## copy till here
    # extract label eigenmodes
    G, label_vertidx, src_flip = _extract_label_eigenmodes(forward, labels, gain.T, mode, n_eigenmodes, allow_empty=True)
    # test if there are empty columns

    sel = np.any(G, axis=0)
    G = G[:, sel].copy()
    label_vertidx = [i for select, i in zip(sel, label_vertidx) if select]
    src_flip = [i for select, i in zip(sel, src_flip) if select]

    _, S, _ = np.linalg.svd(G)
    G /= np.max(np.absolute(np.diag(S)))


    discarded_labels =[]
    j = 0
    for i, sel_ in enumerate(sel[::n_eigenmodes]):
        if not sel_:
            discarded_labels.append(labels.pop(i-j))
            j += 1
    assert j == len(discarded_labels)
    if j > 0:
        logger.info('No sources were found in following {:d} ROIs:\n'.format(len(discarded_labels)) +
                    '\n'.join(map(lambda x: str(x.name), discarded_labels)))


    ROIs_idx = list()

    if ROIs_names == None:
        ROIs_idx = list(range(0, len(labels)))
    else:
        for ROIs_name in ROIs_names:
            ROIs_idx.extend([i for i, label in enumerate(labels) if ROIs_name in label.name])

    out_obj = _nlgc_map_opt(name, M, G, p, n_eigenmodes=n_eigenmodes, ROIs=ROIs_idx, n_segments=n_segments, alpha=alpha, beta=beta,
                            lambda_range=lambda_range, max_iter=max_iter, max_cyclic_iter=max_cyclic_iter, tol=tol, sparsity_factor=sparsity_factor)

    return out_obj


def _nlgc_map_opt(name, M, gain, p, n_eigenmodes=2, ROIs='just_full_model', n_segments=1, alpha=0, beta=0,
                  lambda_range=[1, 0.1, 0.01, 0.001], max_iter=50, max_cyclic_iter=5, tol=1e-8, sparsity_factor=0.1):
    ny, nnx = gain.shape
    nx = nnx // n_eigenmodes
    _, t = M.shape
    tt = t // n_segments

    d_raw = np.zeros((n_segments, nx, nx))
    bias_r = np.zeros((n_segments, nx, nx))
    bias_f = np.zeros((n_segments, 1))
    a_f = np.zeros((n_segments, p, nnx, nnx))
    q_f = np.zeros((n_segments, nnx, nnx))
    lambda_f = np.zeros((n_segments, 1))
    ll_f = np.zeros((n_segments, max_iter))
    conv_flag = np.zeros((n_segments, nx, nx))

    for n in range(0, n_segments):
        print('Segment: ', n+1)
        d_raw_, bias_r_, bias_f_, a_f_, q_f_, lambda_f_, ll_f_, conv_flag_ = \
            gc_extraction(M[:, n*tt: (n+1)*tt].T, gain, p=p, n_eigenmodes=n_eigenmodes, ROIs=ROIs, alpha=alpha, beta=beta,
                          lambda_range=lambda_range, max_iter=max_iter, max_cyclic_iter=max_cyclic_iter, tol=tol, sparsity_factor=sparsity_factor)

        d_raw[n, :, :] = d_raw_
        bias_r[n, :, :] = bias_r_
        bias_f[n] = bias_f_
        a_f[n, :, :, :] = a_f_
        q_f[n, :, :] = q_f_
        lambda_f[n] = lambda_f_
        ll_f[n, 0:len(ll_f_)] = ll_f_
        conv_flag[n, :, :] = conv_flag_


    nlgc_obj = NLGC(name, nx, ny, t, p, n_eigenmodes, n_segments, d_raw, bias_f, bias_r, a_f, q_f, lambda_f, ll_f, conv_flag)

    return nlgc_obj


def _extract_label_eigenmodes(fwd, labels, data=None, mode='mean', n_eigenmodes=2, allow_empty=False,
        trans=None, mri_resolution=True,):
    "Zero columns corresponds to empty labels"
    from mne.source_space import SourceSpaces
    from mne.utils import (logger, _check_option, _validate_type)
    from mne.source_estimate import (_prepare_label_extraction, _BaseSourceEstimate,
                                     _BaseVolSourceEstimate,_BaseVectorSourceEstimate, SourceEstimate,
                                     MixedSourceEstimate, VolSourceEstimate)
    src = fwd['src']
    _validate_type(src, SourceSpaces)
    _check_option('mode', mode, ['svd', 'svd_flip'] + ['auto'])
    func = _svd_funcs[mode]

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

        if data is None:
            logger.info('Using the raw forward solution')
            data = np.swapaxes(fwd['sol']['data'], 0, 1)  # (n_sources, n_channels)
        data = data.copy()

        # do the extraction
        label_eigenmodes = np.zeros((n_labels * n_eigenmodes,) + data.shape[1:], dtype=data.dtype)
        for i, (vertidx, flip, label) in enumerate(zip(label_vertidx, src_flip, labels)):
            if vertidx is not None:
                if isinstance(vertidx, sparse.csr_matrix):
                    assert mri_resolution
                    assert vertidx.shape[1] == data.shape[0]
                    this_data = np.reshape(data, (data.shape[0], -1))
                    this_data = vertidx * this_data
                    this_data.shape = \
                        (this_data.shape[0],) + stc.data.shape[1:]
                else:
                    this_data = data[vertidx]
                label_eigenmodes[i*n_eigenmodes:(i+1)*n_eigenmodes] = \
                    func(flip, this_data, n_eigenmodes)

        return label_eigenmodes.T, label_vertidx, src_flip


def string_link(reg_idx, emod):
    temp = f"{list(range(reg_idx*emod, reg_idx*emod + emod))}"
    temp = temp.replace(" ", "")
    temp = temp.replace("[", "")
    temp = temp.replace("]", "")

    return temp


def gc_extraction(y, f, p, n_eigenmodes=2, ROIs='just_full_model', alpha=0, beta=0, lambda_range=[1, 0.1, 0.01, 0.001], \
                  max_iter=50, max_cyclic_iter=5, tol=1e-8, sparsity_factor=0.1):

    n, m = f.shape
    nx = m // n_eigenmodes

    r = np.eye(n)

    kwargs = {'max_iter': max_iter,
              'max_cyclic_iter': max_cyclic_iter,
              'rel_tol': tol}

    # learn the full model
    model_f = NeuraLVARCV(p, 10, 3, 5, use_lapack=False)


    model_f.fit(y.T, f, r, lambda_range, a_init=None, q_init=np.eye(m), alpha=alpha, beta=beta, **kwargs)

    a_f = model_f._parameters[0]
    q_f = model_f._parameters[2]
    lambda_f = model_f.lambda_
    bias_f = model_f.compute_bias(y.T)

    dev_raw = np.zeros((nx, nx))
    bias_r = np.zeros((nx, nx))
    conv_flag = np.zeros((nx, nx))

    # learn reduced models
    a_init = np.empty_like(a_f)

    if ROIs is None:
        ROIs = range(0, nx)
    elif ROIs == 'just_full_model':
        ROIs = [0]

    sparsity = np.sum(np.absolute(a_f), axis=0)

    for i, j in tqdm(itertools.product(ROIs, repeat=2)):
        if i == j:
            continue
        if np.sum(sparsity[j * n_eigenmodes: (j + 1) * n_eigenmodes, i * n_eigenmodes: (i + 1) * n_eigenmodes]) \
                <= sparsity_factor*np.max(a_f[:, j * n_eigenmodes: (j + 1) * n_eigenmodes, :]):
            continue

        target = string_link(i, n_eigenmodes)
        src = string_link(j, n_eigenmodes)

        link = f"{target}->{src}"
        # "{:s}->{:s}".format(target, src)
        # print(link)
        a_init[:] = a_f[:]
        a_init[:, j * n_eigenmodes: (j + 1) * n_eigenmodes, i * n_eigenmodes: (i + 1) * n_eigenmodes] = 0
        model_r = NeuraLVAR(p, use_lapack=False)
        model_r.fit(y.T, f, r, lambda_f, a_init=a_init, q_init=q_f * 1, restriction=link, alpha=alpha, beta=beta, **kwargs)

        bias_r[j, i] = model_r.compute_bias(y.T)

        dev_raw[j, i] = -2 * (model_r._lls[-1] - model_f._lls[-1])

        conv_flag[j, i] = len(model_r._lls)

    return dev_raw, bias_r, bias_f, a_f, q_f, lambda_f, model_f._lls, conv_flag

class NLGC:
    def __init__(self, subject, nx, ny, t, p, n_eigenmodes, n_segments, d_raw, bias_f, bias_r, a_f, q_f, lambda_f, ll_f, conv_flag):

        self.subject = subject
        self.nx = nx
        self.ny = ny
        self.t = t
        self.p = p
        self.n_eigenmodes = n_eigenmodes
        self.n_segments = n_segments
        self.d_raw = d_raw
        self.bias_f = bias_f
        self.bias_r = bias_r
        self.a_f = a_f
        self.q_f = q_f
        self.lambda_f = lambda_f
        self.ll_f = ll_f
        self.conv_flag = conv_flag

    def plot_ll_curve(self):
        fig, ax = plt.subplots()
        for n in range(0, self.n_segments):
            ll = self.ll_f[n]
            ax.plot(ll[np.nonzero(ll)])

        return fig, ax

    def plot_reduced_models_convergence(self, max_itr=1):
        fig, ax = plt.subplots()
        ax.hist(np.reshape(self.conv_flag/max_itr, (1, self.nx**2)), bins='auto')

        return fig, ax

    def compute_debiased_dev(self):
        d = np.zeros((self.n_segments, self.nx, self.nx))
        for i in range(0, self.n_segments):
            bias_mat = self.bias_r[i].copy()
            bias_mat[bias_mat != 0] -= self.bias_f[i]
            # ipdb.set_trace()

            d[i] = self.d_raw[i]
            d[i][d[i] <= 0] = 0
            d[i] += bias_mat[i]
            d[i][d[i] <= 0] = 0
            np.fill_diagonal(d[i], 0)
        return d

    def fdr(self, alpha=0.1):
        return fdr_control(np.mean(self.compute_debiased_dev(), axis=0), self.p * self.n_eigenmodes, alpha)

    def save_object(self, filename):
        with open(filename+'.obj', 'wb') as filehandler:
            pickle.dump(self, filehandler)

    def plot(self):
        pass

