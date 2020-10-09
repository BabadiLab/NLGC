import mne
import os
import glob
from _nlgc import *
import ipdb
from codetiming import Timer

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
    p = 4
    n_eigenmodes = 2
    n_segments = 1
    max_iter = 100
    max_cyclic_iter = 2
    tol = 1e-4
    sparsity_factor = 0.00
    # ROIs_names = ['superiorfrontal', 'rostralmiddlefrontal', 'caudalmiddlefrontal', 'parsopercularis',
    #               'parstriangularis', 'parsorbitalis', 'caudalanteriorcingulate', 'insula',
    #               'lateralorbitofrontal', # Frontal
    #               'inferiorparietal', 'posteriorcingulate', # Parietal
    #               'superiortemporal', 'middletemporal', 'inferiortemporal', 'transversetemporal'
    #               'bankssts', 'fusiform',] # temporal

    ROIs_names = ['rostralmiddlefrontal', 'caudalmiddlefrontal', 'parsopercularis', 'parstriangularis', 'superiortemporal',
                  'middletemporal', 'transversetemporal']



    # ROIs_names = ['precentral', 'superiorparietal']
    # ROIs_names = ['just_full_model']
    # lambda_range_ = np.array([-8, -9, -10, -11, -12, -13, -14, -15, -16, -18, -20, -22, -24])
    # lambda_range = 1 / (10 ** -lambda_range_)
    lambda_range = None
    lambda_range = [1e3, 1e1, 1e0]
    alpha = 0
    # alpha = 0.1
    beta = 0
    # beta = 0.05 * 0.1
    out = nlgc_map('test', evoked[0], forward, er_cov, labels, p=p, n_eigenmodes=n_eigenmodes, alpha=alpha,
             beta=beta, ROIs_names=ROIs_names, n_segments=n_segments, lambda_range=lambda_range,
             max_iter=max_iter, max_cyclic_iter=max_cyclic_iter, tol=tol, sparsity_factor=sparsity_factor, depth=0.0)

    # obj_list = []
    # with Timer(name='testMEG'):
    #     alpha = 0
    #     beta = 0
    #     obj_list.append(nlgc_map('test', evoked[0], forward, er_cov, labels, p=p, n_eigenmodes=n_eigenmodes, alpha=alpha,
    #                    beta=beta, ROIs_names=ROIs_names, n_segments=n_segments, lambda_range=lambda_range,
    #                    max_iter=max_iter, max_cyclic_iter=max_cyclic_iter, tol=tol, sparsity_factor=sparsity_factor))
    #     # obj_list[0].save_object('o0')
    #
    # print(np.diag(obj_list[0].q_f[0]))
    # print('log-likelihood is:', obj_list[0].ll_f[0])
    # print('lambda is:', obj_list[0].lambda_f[0])

    # with Timer(name='testMEG'):
    #     alpha = 0.1
    #     beta = alpha * np.median(np.diag(obj_list[0].q_f[0]))
    #     obj_list.append(nlgc_map('test', evoked[0], forward, er_cov, labels, p=p, n_eigenmodes=n_eigenmodes, alpha=alpha,
    #                    beta=beta, ROIs_names=ROIs_names, n_segments=n_segments, lambda_range=lambda_range,
    #                    max_iter=max_iter, max_cyclic_iter=max_cyclic_iter, tol=tol, sparsity_factor=sparsity_factor))
    #
    #     obj_list[1].save_object('o1')
    #
    # print(np.diag(obj_list[1].q_f[0]))
    # print('log-likelihood is:', obj_list[1].ll_f[0])
    #
    # with Timer(name='testMEG'):
    #     alpha = 0.2
    #     beta = alpha * np.median(np.diag(obj_list[0].q_f[0]))
    #     obj_list.append(nlgc_map('test', evoked[0], forward, er_cov, labels, p=p, n_eigenmodes=n_eigenmodes, alpha=alpha,
    #                   beta=beta, ROIs_names=ROIs_names, n_segments=n_segments, lambda_range=lambda_range,
    #                   max_iter=max_iter, max_cyclic_iter=max_cyclic_iter, tol=tol, sparsity_factor=sparsity_factor))
    #
    #     obj_list[2].save_object('o2')
    #
    # print(np.diag(obj_list[2].q_f[0]))
    # print('log-likelihood is:', obj_list[2].ll_f[0])


        # lambda_star = o1.lambda_f
        # lambda_range = [lambda_star*10, lambda_star*5, lambda_star, lambda_star/5, lambda_star/10]
        # print('done!')
        # # narrow_down the lambda_range
        # o2 = nlgc_map('test2', evoked[0], forward, er_cov, labels, p=p, n_eigenmodes=n_eigenmodes,
        #               ROIs_names=ROIs_names, n_segments=3, lambda_range=lambda_range,
        #               max_iter=max_iter, max_cyclic_iter=max_cyclic_iter, tol=tol, sparsity_factor=sparsity_factor)


