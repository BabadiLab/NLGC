import mne
import os
import glob
from _nlgc import *
import ipdb
from codetiming import Timer

import warnings
warnings.filterwarnings('error')

## Change this folder name to "'your_mounted_drive'\behrad\Aging" before running the script
# behrad_root = "G:\\My Drive\\behrad\\Aging"
behrad_root = "/Users/behrad/Google Drive/behrad/Aging"

## The corresponding files are alraedy uploaded in Drive
evoked = mne.read_evokeds(os.path.join(behrad_root, 'test', 'R2533-pass_single_M02-ave.fif'))
evoked[0] = evoked[0].filter(0.1, 4.5, phase='minimum')
evoked[0] = evoked[0].decimate(8, offset=3)
evoked[0].crop(tmin=20, tmax=60)
forward = mne.read_forward_solution(os.path.join(behrad_root, 'test', 'R2533-ico-4-fwd.fif'))
er_cov = mne.read_cov(os.path.join(behrad_root, 'test', 'emptyroom-cov.fif'))
fname_labels = os.path.join(behrad_root, 'test', 'labels', 'R2533-*.label')
labels = [mne.read_label(fname_label) for fname_label in glob.glob(fname_labels)]

# run different eigenmodes and compare ll_f
if __name__ == "__main__":

    # filehandler = open('o0.obj', 'rb')
    # obj1 = pickle.load(filehandler)
    order = 4
    p1 = 4
    n_eigenmodes = 2
    n_segments = 1
    max_iter = 100
    max_cyclic_iter = 1
    tol = 1e-4
    sparsity_factor = 0.00

    ROIs_names = ['rostralmiddlefrontal', 'caudalmiddlefrontal', 'parsopercularis', 'parstriangularis', 'superiortemporal',
                  'middletemporal', 'transversetemporal']

    # Inv-Gamma(alpha * 600, beta * 600)

    epsilon = 0.02

    alpha = 0
    beta = 0
    lambda_range = [1e1, 5, 2, 1, 5e-1, 2e-1, 1e-1, 5e-2, 2e-2,]
    lambda_range = [1,]
    out = nlgc_map('test', evoked[0], forward, er_cov, labels, order=order, self_history=p1,
                   n_eigenmodes=n_eigenmodes, alpha=alpha, beta=beta, ROIs_names=ROIs_names, n_segments=n_segments,
                   lambda_range=lambda_range, max_iter=100, max_cyclic_iter=max_cyclic_iter, tol=tol,
                   sparsity_factor=sparsity_factor, depth=0.0)


    from _plot_utils import visualize_con
    d = out.compute_debiased_dev()[0]
    d[d<0] = 0
    im, cbar = visualize_con(d, out._labels, n_eigenmodes=1)