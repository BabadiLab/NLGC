import mne
import os
import glob
from _nlgc import *

## Change this folder name to "'your_mounted_drive'\behrad\Aging" before running the script
behrad_root = r"G:\My Drive\behrad\Aging"

## The corresponding files are alraedy uploaded in Drive
evoked = mne.read_evokeds(os.path.join(behrad_root, 'test', 'R2533-pass_single_M02-ave.fif'))
forward = mne.read_forward_solution(os.path.join(behrad_root, 'test', 'R2533-ico-4-fwd.fif'))
er_cov = mne.read_cov(os.path.join(behrad_root, 'test', 'emptyroom-cov.fif'))
fname_labels = os.path.join(behrad_root, 'test', 'labels', 'R2533-*.label')
labels = [mne.read_label(fname_label) for fname_label in glob.glob(fname_labels)]

out = nlgc_map(evoked, forward, er_cov)

