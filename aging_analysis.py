import mne
import os
import glob
from _nlgc import *
import ipdb
from codetiming import Timer

import warnings
warnings.filterwarnings('ignore')

from csslaging import e
from csslaging import er_cov as cov

from matplotlib import pyplot as plt


if __name__ == "__main__":

    subject = 'R2534'

    e.set(subject=subject)
    ds = e.load_epochs(subject, decim=20, ndvar=False, reject=False)
    labels = e._load_labels()
    labels_as_list = list(labels.values())
    forward = e.load_fwd(ndvar=False)

    epochs = ds['epochs'][ds['noise'] == '-6dB']
    # epochs = ds['epochs'][ds['condition'] == 'quiet']

    evoked = epochs[0].average()
    evoked = evoked.filter(None, 4.5, phase='minimum')
    evoked = evoked.decimate(8, offset=3)
    evoked.crop(tmin=20, tmax=60)

    # filehandler = open('o0.obj', 'rb')
    # obj1 = pickle.load(filehandler)
    order = 4
    p1 = 4
    n_eigenmodes = 2
    n_segments = 1
    max_iter = 1000
    max_cyclic_iter = 1
    tol = 1e-8
    sparsity_factor = 0.0

    ROIs_names = ['rostralmiddlefrontal', 'caudalmiddlefrontal', 'parsopercularis', 'parstriangularis',
                  'superiortemporal',
                  'middletemporal', 'transversetemporal']

    alpha = 0
    beta = 0
    lambda_range = [1e2, 1e1, 5e0, 2e0, 1e0, 5e-1, 2e-1, 1e-1, 5e-2, 2e-2, 1e-2]
    lambda_range = [1e2, 1e1, 5e0, 2e0, 1e0, 5e-1]
    lambda_range = [5, 3, 2, 1.7, 1, 0.5, 0.2]
    lambda_range = [2]

    # -6db                 |           quiet
    #R2533 -> 0.24         |            0.55
    #R2534 -> 2.09         |            1.75   ???
    #R2654 -> 2.45         |            2.65   ???
    #R2553 -> 1.09         |            0.51   ???

    out = nlgc_map(subject, evoked, forward, cov, labels_as_list, order=order, self_history=p1,
                   n_eigenmodes=n_eigenmodes, alpha=alpha, beta=beta, ROIs_names=ROIs_names, n_segments=n_segments,
                   lambda_range=lambda_range, max_iter=max_iter, max_cyclic_iter=max_cyclic_iter, tol=tol,
                   sparsity_factor=sparsity_factor, depth=0.0)

    from _plot_utils import visualize_con



    import warnings
    warnings.filterwarnings('ignore')

    d = out.compute_debiased_dev()[0]
    fig, ax = plt.subplots()
    im, cbar = visualize_con(d, out._labels, n_eigenmodes=1, ax=ax)

    fig.show()

    # with open('R2553-clean.pkl', 'wb') as fp:
    #     pickle.dump(out, fp)
    #
    # with open('R2533-6dB.pkl', 'rb') as fp:
    #     out=pickle.load(fp)