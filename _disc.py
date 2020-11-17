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

import matplotlib
import matplotlib.pyplot as plt

# kwargs = {'raw': 'tsss-1-8-causal-ica-apply', 'src': 'ico-4', 'parc': 'aparc', 'epoch': 'all', 'session': 'Cocktail'}
# e.set(**kwargs)
# kwargs = {'src': 'ico-4'}
# e.set(**kwargs)

class DiagFls:
    def __init__(self, m_active, m_inactive, JG, J, patch_idx):
        self.m_active = m_active
        self.m_inactive = m_inactive
        self.JG = JG*1
        self.J = J
        self.patch_idx = patch_idx

    def print_links(self, trial):
        patch = self.patch_idx[trial]
        print('############################################################')
        print('true links of trial ', trial, ' :')
        for tar, src in zip(np.nonzero(self.JG)[0], np.nonzero(self.JG)[1]):
            print(patch[src], '->', patch[tar])
        print('------------------------------------------------------------')
        print('discovered links of trial ', trial, ' :')
        for tar, src in zip(np.nonzero(self.J[trial])[0], np.nonzero(self.J[trial])[1]):
            print(patch[src], '->', patch[tar])
        print('############################################################')

def check_labels(temp, alab, ROI_names, trial):
    patch = temp.patch_idx[trial]
    J = temp.J[trial]
    JG = temp.JG
    print('############################################################')
    print('true links of trial ', trial, ' :')
    for tar, src in zip(np.nonzero(JG)[0], np.nonzero(JG)[1]):
        print(return_label(patch[src], ROI_names, alab), '->', return_label(patch[tar], ROI_names, alab))
    print('------------------------------------------------------------')
    print('discovered links of trial ', trial, ' :')
    for tar, src in zip(np.nonzero(J)[0], np.nonzero(J)[1]):
        print(return_label(patch[src], ROI_names, alab), '->', return_label(patch[tar], ROI_names, alab))
    print('############################################################')

def return_label(patch_idx, ROI_names, alab):
    for i in range(len(alab)):
        if patch_idx in alab[i]:
            ROI_idx = i
    return ROI_names[ROI_idx]


if __name__ == "__main__":

    np.random.seed(0)

    ds = e.load_epochs(ndvar=False, reject=False)
    labels = e._load_labels()
    labels_as_list = list(labels.values())
    forward = e.load_fwd(src='ico-4', ndvar=False)
    src_origin = forward['src']

    epochs = ds['epochs'][ds['noise'] == '-6dB']
    evoked = epochs[0].average()


    src_target = e.load_src(src='ico-1')
    grouped_vertidx, n_groups, n_verts = _prepare_leadfield_reduction(src_target, src_origin)
    labels_as_list = src_target

    alab = assign_labels(list(labels.values()), src_target, src_origin)

    # file_name = f"{m_active}-{m_inactive}-{alpha}.pkl"
    file_name = f"{4}-{4}-{0.5}.pkl"
    with open(file_name, 'rb') as fp: temp = pickle.load(fp)

    ROI_names = list(labels.keys())
    trial = 10
    check_labels(temp, alab, ROI_names, trial)





