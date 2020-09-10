import numpy as np
import copy
import pickle


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
        rank=None, mode='svd_flip', n_eigenmodes=2,):
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

    X = simulate_AR_process()
    G = _undo_source_weighting(gain, source_weighting)

    label_vertidxs, src_flip = _extract_label_eigenmodes(forward, labels)

    # get the data
    Y = forward_mapping(G, label_vertidxs, src_flip, X)
    Y = Y + noise  # simluate noise ~ N(0, 1) already prewhitened

    # whiten the data
    logger.info('Whitening data matrix.')
    # No need
    # save the data
    pickle.dump(Y, 'sim_data.pickled')
    return Y


def run_experiment():