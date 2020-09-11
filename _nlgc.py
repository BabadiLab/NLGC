# author: proloy Das <proloy@umd.edu>

import numpy as np
from scipy import linalg, sparse
import copy

from mne.forward import is_fixed_orient

from mne.minimum_norm.inverse import _check_reference
from mne.utils import logger, verbose, warn
from mne.inverse_sparse.mxne_inverse import _prepare_gain


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


def nlgc_map(evoked, forward, noise_cov, labels, loose=0.0, depth=0.8, maxit=10000, tol=1e-6, pca=True,
        rank=None, mode='svd_flip', n_eigenmodes=2,):
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
    ## copy till here
    # extract label eigenmodes
    G = _extract_label_eigenmodes(forward, labels, gain.T, mode, n_eigenmodes, allow_empty=True)
    # test if there are empty columns
    sel = np.any(G, axis=0)
    G = G[:, sel].copy()
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

    # TODO run the optimization
    import ipdb
    ipdb.set_trace()
    outs  = _nlgc_map_opt(M, G, maxit=maxit, tol=tol, n_eigenmodes=n_eigenmodes)

    # TODO Compute the necessary things, and save as an object
    out_obj = 0

    return out_obj


def _nlgc_map_opt(M, gain, maxit, tol, n_eigenmodes):
    #TODO implement
    raise NotImplementedError


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
