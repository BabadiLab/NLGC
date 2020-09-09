import mne
import os
import glob

behrad_root = r"G:\My Drive\behrad\Aging"

evoked = mne.read_evokeds(os.path.join(behrad_root, 'test', 'R2533-pass_single_M02-ave.fif'))
forward = mne.read_forward_solution(os.path.join(behrad_root, 'test', 'R2533-ico-4-fwd.fif'))
er_cov = mne.read_cov(os.path.join(behrad_root, 'test', 'emptyroom-cov.fif'))
fname_labels = os.path.join(behrad_root, 'test', 'labels', 'R2533-*.label')
labels = [mne.read_label(fname_label) for fname_label in glob.glob(fname_labels)]