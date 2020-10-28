# author: proloy Das <proloy@umd.edu>
# author: Behrad Soleimani <behrad@umd.edu>

import ipdb
import itertools
from nlgc.opt.opt import NeuraLVAR, NeuraLVARCV, NeuraLVARCV_
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy import linalg, sparse
import copy
from mne import (Forward, Label, SourceSpaces)
from mne.forward import is_fixed_orient
from mne.minimum_norm.inverse import _check_reference
from mne.utils import logger, verbose, warn
from mne.inverse_sparse.mxne_inverse import _prepare_gain
import ipdb
from nlgc._stat import fdr_control
import pickle
# from nlgc.core import gc_extraction, NLGC
from codetiming import Timer
from multiprocessing import cpu_count

import warnings

plt.ion()

def truncatedsvd(a, n_components=2, return_pecentage_exaplained=False):
    if n_components > min(*a.shape):
        raise ValueError('n_components={:d} should be smaller than '
                         'min({:d}, {:d})'.format(n_components, *a.shape))
    u, s, vh = linalg.svd(a, full_matrices=False, compute_uv=True,
                          overwrite_a=True, check_finite=True,
                          lapack_driver='gesdd')
    if return_pecentage_exaplained:
        return vh[:n_components] * s[:n_components][:, None], s[:n_components].sum() / s.sum()
    return vh[:n_components] * s[:n_components][:, None]


_svd_funcs = {
    'svd_flip': lambda flip, data, n_components: truncatedsvd(flip * data, n_components),
    'svd': lambda flip, data, n_components: truncatedsvd(data, n_components)
}


# Note for covariance, source_weighting needs to be applied twice!
def _reapply_source_weighting(X, source_weighting):
    X *= source_weighting[:, None]
    return X


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


def expand_roi_indices_as_tup(reg_idx, emod):
    return tuple(range(reg_idx * emod, reg_idx * emod + emod))


_default_lambda_range = [5e-1, 2e-1, 1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4,]

def _gc_extraction(y, f, r, p, p1, n_eigenmodes=2, ROIs='just_full_model', alpha=0, beta=0,
                  lambda_range=None, max_iter=50, max_cyclic_iter=5,
                  tol=1e-3, sparsity_factor=0.0, cv=5, use_lapack=True):

    n, m = f.shape
    nx = m // n_eigenmodes

    kwargs = {'max_iter': max_iter,
              'max_cyclic_iter': max_cyclic_iter,
              'rel_tol': tol}

    # learn the full model
    n_jobs = cv if isinstance(cv, int) else cv.get_n_splits()
    n_jobs = min(n_jobs, cpu_count())
    # n_jobs = 3

    if lambda_range is None:
        lambda_range = _default_lambda_range

    e, u = linalg.eigh(f.dot(f.T))
    temp = u.T.dot(y)
    c = (temp ** 2).sum(axis=1)
    from scipy import optimize
    fun = lambda x:  (c / (1 + x * e)**2).sum() - 1.2 * n * y.shape[1]
    fprime = lambda x: - 2 * ((c * e) / (1 + x * e)**3).sum()

    if fun(0) > 0:
        q_val = optimize.newton(fun, 1)
    else:
        q_val = 0.0001
    q_init = q_val * np.eye(m)
    a_init = None

    # q_init = 1 * np.eye(m)
    # a_init = None
    # dummy_model_f = NeuraLVAR(p, p1, use_lapack=use_lapack)
    # dummy_model_f.fit(y, f, r * np.eye(n), lambda_range[0], a_init=None, q_init=q_init, restriction=None, alpha=alpha,
    #                   beta=beta, **kwargs)
    # a_init = dummy_model_f._parameters[0]
    # q_init = dummy_model_f._parameters[2]

    # sol = linalg.lstsq(f, y)
    # x_est = sol[0]
    # x_ = np.vstack(tuple([x_est[:, p-i:-i] for i in range(1,p+1)]))
    # s1 = x_est[:, p:].dot(x_.T) / x_.shape[-1]
    # s2 = x_.dot(x_.T) / x_.shape[-1]
    # lambda2 = np.abs(s1).max() / (10 * 1)
    # a_init = np.zeros_like(s1)
    # from nlgc.opt.m_step import solve_for_a, solve_for_q
    # a_init, _ = solve_for_a(q_init, s1, s2, a_init, p1, lambda2)
    #
    # a_init = np.swapaxes(np.reshape(a_init, (m, p, m)), 0, 1)

    if len(lambda_range) > 1:
        model_f = NeuraLVARCV_(p, p1, 10, cv, n_jobs, use_lapack=use_lapack)
    else:
        model_f = NeuraLVAR(p, p1, use_lapack=use_lapack)
        lambda_range = lambda_range[0]
    model_f.fit(y, f, r * np.eye(n), lambda_range, a_init=a_init, q_init=q_init.copy(), alpha=alpha, beta=beta,
                **kwargs)
    bias_f = model_f.compute_bias(y)

    warnings.filterwarnings('ignore')
    model_f.compute_crossvalidation_metric(y)
    # ipdb.set_trace()

    dev_raw = np.zeros((nx, nx))
    bias_r = np.zeros((nx, nx))
    conv_flag = np.zeros((nx, nx), dtype=np.bool_)

    # learn reduced models
    a_f = model_f._parameters[0]
    q_f = model_f._parameters[2]
    lambda_f = model_f.lambda_
    a_init = np.empty_like(a_f)
    mul = np.exp(n_eigenmodes * p / y.shape[1])

    (a_f, q_f, bias_r, dev_raw, conv_flag, y, f)  # shared memory
    (r, alpha, beta, n_eigenmodes, kwargs) # can be passed directly

    sparsity = np.linalg.norm(model_f._parameters[0], axis=0, ord=1) * np.diag(model_f._parameters[2])[None, :]

    bias_r_ = np.zeros((nx, nx))
    bias_f_ = np.zeros((nx, nx))
    dev_raw_ = np.zeros((nx, nx))
    for i, j in tqdm(itertools.product(ROIs, repeat=2)):
        if i == j:
            continue

        target = expand_roi_indices_as_tup(j, n_eigenmodes)
        source = expand_roi_indices_as_tup(i, n_eigenmodes)
        if sparsity[target, source].sum() <= sparsity_factor * sparsity[target, target].sum():
            continue

        link = '->'.join(map(lambda x: ','.join(map(str, x)), (source, target)))
        a_init[:] = a_f[:]
        a_init[:, target, source] = 0.0
        model_r = NeuraLVAR(p, p1, use_lapack=use_lapack)
        model_r.fit(y, f, r*np.eye(n), lambda_f, a_init=None, q_init=q_init.copy(), restriction=link,
                    alpha=alpha,
                    beta=beta, **kwargs)
        # model_r = NeuraLVARCV_(p, p1, 10, cv, n_jobs, use_lapack=use_lapack)
        # model_r.fit(y, f, r*np.eye(n), lambda_range, a_init=a_init.copy(), q_init=q_init.copy(), restriction=link,
        #             alpha=alpha,
        #             beta=beta, **kwargs)
        # print(model_r._lls)
        # warnings.filterwarnings('ignore')
        # ipdb.set_trace()


        bias_r[j, i] = model_r.compute_bias(y)

        dev_raw[j, i] = -2 * (model_r.ll - model_f.ll)

        dev_raw_[j, i] = sum(map(lambda x: np.log(model_r._parameters[2][x, x]) - np.log(model_f._parameters[2][x, x]),
                                 target))
        # if dev_raw_[j, i] < 0:
        #     warnings.filterwarnings('ignore')
        #     ipdb.set_trace()
        bias_r_[j, i] = model_r.compute_bias_idx(y, target)
        bias_f_[j, i] = model_f.compute_bias_idx(y, target)

        # import ipdb;ipdb.set_trace()
        conv_flag[j, i] = len(model_r._lls[0]) == max_iter

    dev_raw_ *= (y.shape[1] - p)
    return dev_raw, bias_r, bias_f, model_f, conv_flag, dev_raw_, bias_f_, bias_r_


class NLGC:
    def __init__(self, subject, nx, ny, t, p, n_eigenmodes, n_segments, d_raw, bias_f, bias_r,
            model_f, conv_flag, label_names, label_vertidx, debug=None):

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
        self._model_f = model_f
        self._conv_flag = conv_flag
        self._labels = label_names
        self._label_vertidx = label_vertidx
        self._debug = debug

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
        d = self.d_raw.copy()
        for i in range(0, self.n_segments):
            bias_mat = self.bias_r[i].copy()
            bias_mat[bias_mat != 0] -= self.bias_f[i]

            d[i] += bias_mat
            np.fill_diagonal(d[i], 0)
        return d

    def fdr(self, alpha=0.1):
        return fdr_control(np.mean(self.compute_debiased_dev(), axis=0), self.p * self.n_eigenmodes, alpha)

    def save_object(self, filename):
        with open(filename+'.obj', 'wb') as filehandler:
            pickle.dump(self, filehandler)

    def plot(self):
        pass


def _nlgc_map_opt(name, M, gain, r, p, p1, n_eigenmodes=2, ROIs='just_full_model', n_segments=1, alpha=0, beta=0,
                  lambda_range=None, max_iter=50, max_cyclic_iter=5, tol=1e-3, sparsity_factor=0.1,
                  cv=5, label_names=None, label_vertidx=None, use_lapack=True):
    ny, nnx = gain.shape
    nx = nnx // n_eigenmodes
    _, t = M.shape
    tt = t // n_segments

    d_raw = np.zeros((n_segments, nx, nx))
    bias_r = np.zeros((n_segments, nx, nx))
    bias_f = np.zeros((n_segments, 1))
    conv_flag = np.zeros((n_segments, nx, nx))
    _d_raws = []
    _bias_rs = []
    _bias_fs = []
    models = []

    for n in range(0, n_segments):
        print('Segment: ', n+1)
        d_raw_, bias_r_, bias_f_, model_f, conv_flag_, _d_raw, _bias_f, _bias_r = \
            _gc_extraction(M[:, n * tt: (n + 1) * tt], gain, r, p=p, p1=p1, n_eigenmodes=n_eigenmodes, ROIs=ROIs,
                           alpha=alpha, beta=beta, cv=cv, lambda_range=lambda_range, max_iter=max_iter,
                           max_cyclic_iter=max_cyclic_iter, tol=tol, sparsity_factor=sparsity_factor,
                           use_lapack=use_lapack)

        d_raw[n] = d_raw_
        bias_r[n] = bias_r_
        bias_f[n] = bias_f_
        _d_raws.append(_d_raw)
        _bias_fs.append(_bias_f)
        _bias_rs.append(_bias_r)
        models.append(model_f)
        conv_flag[n] = conv_flag_


    nlgc_obj = NLGC(name, nx, ny, t, p, n_eigenmodes, n_segments, d_raw, bias_f, bias_r, models,
                    conv_flag, label_names, label_vertidx, (_d_raws, _bias_fs, _bias_rs))

    return nlgc_obj


def nlgc_map(name, evoked, forward, noise_cov, labels, order, self_history=None, n_eigenmodes=2, alpha=0, beta=0,
        ROIs_names='just_full_model',
        n_segments=1, loose=0.0, depth=0.8, pca=True, rank=None, mode='svd_flip', lambda_range=None, max_iter=50,
        max_cyclic_iter=5, tol=1e-3, sparsity_factor=0.0, cv=5, use_lapack=True):
    _check_reference(evoked)

    depth_dict={'exp':depth, 'limit_depth_chs':'whiten', 'combine_xyz':'fro', 'limit':None}

    forward, gain, gain_info, whitener, source_weighting, mask = _prepare_gain(
        forward, evoked.info, noise_cov, pca, depth_dict, loose, rank)

    if not is_fixed_orient(forward):
        raise ValueError(f"Cannot work with free orientation forward: {forward}")

    # get the data
    sel = [evoked.ch_names.index(name) for name in gain_info['ch_names']]
    M = evoked.data[sel]

    # whiten the data
    logger.info('Whitening data matrix.')
    M = np.dot(whitener, M)
    ## copy till here
    # extract label eigenmodes
    # G, label_vertidx, src_flip = _extract_label_eigenmodes(forward, labels, gain.T, mode, n_eigenmodes, allow_empty=True)
    if isinstance(labels, Forward):
        G, label_vertidx, src_flip = reduce_lead_field(forward, labels, n_eigenmodes, data=gain.T)
        label_names = []
        for label in labels['src']:
            label_names.extend(label['vertno'])
    elif isinstance(labels, SourceSpaces):
        G, label_vertidx, src_flip = reduce_lead_field(forward, labels, n_eigenmodes, data=gain.T)
        label_names = []
        for label in labels: label_names.extend(label['vertno'])
    elif isinstance(labels, list):
        if isinstance(labels[0], Label):
            G, label_vertidx, src_flip = _extract_label_eigenmodes(forward, labels, gain.T, mode, n_eigenmodes,
                                                                   allow_empty=True)
            label_names = [label.name for label in labels]
        else:
            raise ValueError('Not supported {labels}: labels are expected to be either an mne.SourceSpace or'
                             'mne.Forward object or list of mne.Labels.')

    # test if there are empty columns
    sel = np.any(G, axis=0)
    G = G[:, sel].copy()
    label_vertidx = [i for select, i in zip(sel, label_vertidx) if select]
    src_flip = [i for select, i in zip(sel, src_flip) if select]
    discarded_labels =[]
    j = 0
    for i, sel_ in enumerate(sel[::n_eigenmodes]):
        if not sel_:
            discarded_labels.append(labels.pop(i-j))
            label_vertidx.pop(i-j)
            j += 1
    assert j == len(discarded_labels)
    if j > 0:
        logger.info('No sources were found in following {:d} ROIs:\n'.format(len(discarded_labels)) +
                    '\n'.join(map(lambda x: str(x.name), discarded_labels)))

    ROIs_idx = list()
    ipdb.set_trace()

    if ROIs_names == None:
        ROIs_idx = list(range(0, len(labels)))
    elif ROIs_names == 'just_full_model':
        ROIs_idx = []
    else:
        for ROIs_name in ROIs_names:
            ROIs_idx.extend([i for i, label in enumerate(labels) if ROIs_name in label.name])

    # Normalization
    M_normalizing_factor = linalg.norm(np.dot(M, M.T)/M.shape[1], ord='fro')
    G_normalizing_factor = linalg.norm(G, ord=2)
    # G /= G_normalizing_factor
    G_normalizing_factor = np.sqrt(np.sum(G ** 2, axis=0))
    G /= G_normalizing_factor[None, :]
    G *= np.sqrt(M_normalizing_factor)
    r = 1
    fig, ax = plt.subplots()
    data = G.T.dot(G) / M_normalizing_factor
    vmax = max(data.max(), -data.min())
    im = ax.matshow(data, vmax=vmax, vmin=-vmax, cmap='seismic')
    ipdb.set_trace()


    out_obj = _nlgc_map_opt(name, M, G, r, order, self_history, n_eigenmodes=n_eigenmodes, ROIs=ROIs_idx, n_segments=n_segments,
                            alpha=alpha, beta=beta,
                            lambda_range=lambda_range, max_iter=max_iter, max_cyclic_iter=max_cyclic_iter, tol=tol,
                            sparsity_factor=sparsity_factor, cv=cv, label_names=label_names,
                            label_vertidx=label_vertidx, use_lapack=use_lapack)

    return out_obj



def reduce_lead_field(forward, src, n_eigenmodes, data=None):
    import mne
    if data is None:
        logger.info('Using the raw forward solution')
        data = np.swapaxes(forward['sol']['data'], 0, 1)  # (n_sources, n_channels)
    data = data.copy()

    if isinstance(src, mne.Forward):
        src = src['src']

    grouped_vertidx, n_groups, n_verts = _prepare_leadfield_reduction(src, forward['src'])
    group_eigenmodes = np.zeros((sum(n_groups) * n_eigenmodes,) + data.shape[1:], dtype=data.dtype)
    for i, this_grouped_vertidx in enumerate(grouped_vertidx):
        this_group_eigenmodes, percentage_explained = truncatedsvd(data[this_grouped_vertidx],
                                                             n_eigenmodes, return_pecentage_exaplained=True)
        group_eigenmodes[i * n_eigenmodes:(i + 1) * n_eigenmodes] = this_group_eigenmodes

    src_flips = [None] * sum(n_groups)
    return group_eigenmodes.T, grouped_vertidx, src_flips


def prepare_label_extraction(labels, src):
    vertno = [s['vertno'] for s in src]
    label_vertidx = []
    for label in labels:
        if label.hemi == 'lh':
            this_vertices = np.intersect1d(vertno[0], label.vertices)
            vertidx = np.searchsorted(vertno[0], this_vertices)
        elif label.hemi == 'rh':
            this_vertices = np.intersect1d(vertno[1], label.vertices)
            vertidx = len(vertno[0]) + np.searchsorted(vertno[1], this_vertices)
        if len(vertidx) == 0:
            vertidx = None
        label_vertidx.append(vertidx)
    return label_vertidx


# def assign_labels(labels, src_target, src_origin):
#     label_vertidx = prepare_label_extraction(labels, src_origin)
#     vertno = [s['vertno'] for s in src_target]
#     label_vertidx = []
#     for label in labels:
#         if label.hemi == 'lh':
#             this_vertices = np.intersect1d(vertno[0], label.vertices)
#             vertidx = np.searchsorted(vertno[0], this_vertices)
#         elif label.hemi == 'rh':
#             this_vertices = np.intersect1d(vertno[1], label.vertices)
#             vertidx = len(vertno[0]) + np.searchsorted(vertno[1], this_vertices)
#         if len(vertidx) == 0:
#             vertidx = None
#         label_vertidx.append(vertidx)
#     return label_vertidx


def _prepare_leadfield_reduction(src_target, src_origin):
    vertno_origin = [s['vertno'] for s in src_origin]
    vertno_target = [s['vertno'] for s in src_target]
    pinfo_target = [s['pinfo'] for s in src_target]
    n_verts = [s['nuse'] for s in src_origin]
    n_groups = [s['nuse'] for s in src_target]
    grouped_vertidx = []
    for k, (this_vertno_target, this_pinfo_target, this_vertno_origin) in enumerate(zip(vertno_target, pinfo_target,
                                                                            vertno_origin)):
        offset = 0 if k == 0 else n_verts[k - 1]
        for this_vert, this_pinfo in zip(this_vertno_target, this_pinfo_target):
            this_vertices = np.intersect1d(this_vertno_origin, this_pinfo)
            vertidx = offset + np.searchsorted(this_vertno_origin, this_vertices)
            if len(vertidx) == 0:
                vertidx = None
            grouped_vertidx.append(vertidx)
    return grouped_vertidx, n_groups, n_verts


def old_bias(model, reg_idx, n_eigenmodes=1):
    """Computes the bias in the deviance (behrad@umd.edu)

    Parameters
    ----------
    model:  the NEURALVAR model
    reg_idx:  corresonding ROI index
    n_eigenmodes:  number of eigenmodes

    Returns
    -------
    bias
    """

    a = np.hstack(model._parameters[0])
    q = model._parameters[2]
    x_bar = model._parameters[4]

    t, dxm = x_bar.shape
    _, dtot = a.shape
    p = dtot // dxm

    bias = 0
    for idx_src in range(reg_idx*n_eigenmodes, (reg_idx+1)*n_eigenmodes):
        ai = a[idx_src]

        xi = x_bar[p:, idx_src]

        cx = np.zeros((t-p, dtot))

        for k in range(p):
            cx[:, k * dxm:(k + 1) * dxm] = x_bar[p - 1 - k:t - 1 - k]

        qd = np.diag(q)

        # gradient of log - likelihood
        ldot = 1 / qd[idx_src] * cx.T @ (xi - cx @ ai)

        # hessian of log - likelihood
        ldotdot = -1 / qd[idx_src] * (cx.T @ cx)

        bias += ldot.T@np.linalg.solve(ldotdot, ldot)

    return bias


def old_log_likelihood(model, reg_idx, n_eigenmodes=1):
    """Computes the src log-likelihood (behrad@umd.edu)

    Parameters
    ----------
    model:  the NEURALVAR model
    reg_idx:  corresonding ROI index
    n_eigenmodes:  number of eigenmodes

    Returns
    -------
    src ll
    """

    a = np.hstack(model._parameters[0])
    q = model._parameters[2]
    x_bar = model._parameters[4]

    t, dxm = x_bar.shape
    _, dtot = a.shape
    p = dtot // dxm  # what's this??

    ll = 0
    for idx_src in range(reg_idx*n_eigenmodes, (reg_idx+1)*n_eigenmodes):
        ai = a[idx_src]

        xi = x_bar[p:, idx_src]

        cx = np.zeros((t-p, dtot))

        for k in range(p):
            cx[:, k * dxm:(k + 1) * dxm] = x_bar[p - 1 - k:t - 1 - k]

        qd_ = np.diag(q)
        qd = qd_[idx_src]

        # ll += -t*np.log(qd)/2 - (np.linalg.norm(xi - cx @ ai)**2)/(2*qd)
        ll += -t*np.log(qd)/2 - t/2


    return ll

