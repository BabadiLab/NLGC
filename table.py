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


def data_generation(patch_idx, m_active, alpha, evoked, forward, cov, labels_as_list, n_eigenmodes):
    if alpha == -1:
        n2 = 10
        W = np.hstack((np.random.uniform(0.75,1.25,(1,2)),np.random.uniform(0,np.sqrt(2)/2,(1,8))))
    else:
        n2 = 3
        W = np.array([1, 1, alpha])

    G, _ = nlgc_map_(evoked, forward, cov, labels_as_list, n_eigenmodes=n2)
    # G1, _ = nlgc_map_(evoked, forward, cov, labels_as_list, n_eigenmodes=n2)
    # G1, _ = nlgc_map_(evoked, forward, cov, labels_as_list, n_eigenmodes=10)

    n, _ = G.shape
    g = np.zeros((n, len(patch_idx)))
    ex_g = np.zeros((n, n_eigenmodes * len(patch_idx)))

    for idx, this_patch in enumerate(patch_idx):
        # patch_vert_idx = grouped_vertidx[this_patch]
        # g_ = gain[:, patch_vert_idx]

        ex_g[:, idx * n_eigenmodes: (idx + 1) * n_eigenmodes] = G[:, this_patch * n2: this_patch * n2 + n_eigenmodes]
        g_ = G[:, this_patch * n2: (this_patch + 1) * n2]

        g0 = np.sum(g_ * W, axis=1)
        g0 /= np.sum(W)
        g[:, idx] = g0 / np.linalg.norm(g0, ord=2)


    ex_g /= np.sqrt(np.sum(ex_g ** 2, axis=0))

    # Simulate AR process
    ###############################################################################

    m, p, t = len(patch_idx), 2, 3000

    q = m_active/(20-m_active)*np.eye(m)

    a = np.zeros(p * m * m, dtype=np.float64)
    a.shape = (p, m, m)

    for i in range(m_active):
        q[i, i] = 1
        a[1, i, i] = 0.9

    if m_active == 2:
        a[0, 0, 1] = -0.5

    if m_active == 3:
        a[0, 0, 1] = -0.5
        a[0, 0, 2] = -0.5

    if m_active == 4:
        a[0, 0, 1] = -0.5
        a[0, 0, 2] = -0.5
        a[0, 3, 2] = -0.5

    if m_active == 5:
        a[0, 0, 1] = -0.5
        a[0, 0, 2] = -0.5
        a[0, 3, 2] = -0.5
        a[0, 3, 4] = -0.5

    # a[0, 2, 2] = 0.8
    # a[0, 3, 3] = 0.8
    # a[0, 4, 4] = -0.8
    # a[0, 5, 5] = -0.8
    # a[0, 6, 6] = -0.8

    # a[1, 0, 4] = -0.4
    # a[0, 1, 5] = 0.4
    # a[1, 3, 1] = 0.4

    #################################################################

    # a[0, 0, 1] = -0.1
    # a[1, 0, 7] = -0.3
    #
    # a[0, 1, 0] = -0.2
    # a[0, 1, 3] = 0.3
    # a[0, 1, 4] = -0.2
    #
    # a[1, 2, 7] = -0.35
    #
    # a[1, 4, 6] = 0.3
    #
    # a[1, 5, 4] = -0.3
    # a[2, 5, 6] = 0.3
    #
    # a[2, 8, 9] = 0.2
    # a[2, 9, 8] = -0.2
    #
    # a[3, 10, 12] = -0.25
    # a[3, 11, 10] = 0.25
    # a[3, 12, 11] = -0.25
    #
    # a[1, 13, 14] = -0.1
    # a[1, 14, 13] = 0.15
    #
    # a[0, 0, 0] = 0.8
    # a[1, 1, 1] = 0.9
    # a[1, 5, 5] = 0.45
    # a[1, 6, 6] = -0.45
    # a[2, 8, 8] = -0.6
    # a[2, 9, 9] = 0.5
    # a[2, 13, 13] = 0.65
    # a[2, 14, 14] = 0.65

    #################################################################

    # a[0, 0, 0] = 0.8
    # a[0, 1, 1] = 0.8
    # a[0, 2, 2] = 0.8
    # a[0, 3, 3] = 0.8
    # a[0, 4, 4] = -0.8
    # a[0, 5, 5] = -0.8
    # a[0, 6, 6] = -0.8
    #
    # a[0, 0, 4] = -0.4
    # a[0, 1, 5] = 0.4
    # a[0, 3, 1] = 0.4

    # a[1, 0, 1] = -0.4
    # a[1, 1, 4] = 0.4
    # a[1, 6, 4] = 0.5

    # a[2, 2, 3] = -0.6
    # a[2, 2, 6] = -0.5

    # a[3, 5, 6] = -0.4
    # a[3, 0, 5] = -0.5

    temp_JG = np.sum(np.abs(a), axis=0)
    JG = temp_JG != 0
    np.fill_diagonal(JG, 0)


    u = np.random.standard_normal(m * t)
    u.shape = (t, m)
    l = linalg.cholesky(q, lower=True)
    u = u.dot(l.T)

    x = np.empty((t, m), dtype=np.float64)
    for i in range(p):
        x[i] = 0.0

    for i in range(p, t):
        x[i] = u[i]
        for k in range(p):
            x[i] += a[k].dot(x[i - k - 1])

    ## Data generation
    ###############################################################################

    y = x.dot(g.T)
    # reduced_ex_g = ex_g[:, ::n_eigenmodes]
    # y = x.dot(reduced_ex_g.T)
    px = y.dot(y.T).trace()

    noise = np.random.standard_normal(y.shape)
    pn = noise.dot(noise.T).trace()
    multiplier = 1e2 * pn / px

    y += noise / np.sqrt(multiplier)
    r_cov = 1 / multiplier

    return ex_g, g, x, y, r_cov, p, JG


def debiased_dev(dev_raw, bias_f, bias_r):
    d = dev_raw.copy()
    bias_mat = bias_r - bias_f

    d[d < 0] = 0
    d[d > 0] += bias_mat[d > 0]
    np.fill_diagonal(d, 0)
    d[d < 0] = 0
    return d

def missed_false_detection(J, J_est):

    n, _ = J.shape
    Jr = J.copy()
    Jr = Jr.reshape((n**2, 1))

    Jr_est = J_est.copy()
    Jr_est = Jr_est.reshape((n**2, 1))
    Jr_est = np.absolute(Jr_est)
    Jr_est[Jr_est != 0] = 1

    missed_det = np.sum(Jr) - np.sum(Jr_est*Jr)
    false_det = np.sum((Jr-Jr_est) == -1)

    return missed_det, false_det


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

if __name__ == "__main__":

    np.random.seed(1)

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

    total_trial = 20
    n_eigenmodes = 2
    # lambda_range = np.asanyarray([2, 1.5, 1.25, 1.1, 1, 0.9, 0.75, 5e-1, 2e-1, 1e-1, 1e-2])
    lambda_range = np.asanyarray([1.5, 1.25, 1.1, 1, 0.9, 0.75, 5e-1, 2e-1, 1e-1])
    max_iter = 500
    max_cyclic_iter = 3
    tol = 1e-5
    sparsity_factor = 0.0
    n_segments = 3

    # 'transversetemporal-lh':      19, 25, 28
    # 'transversetemporal-rh':      58, 67
    # 'superiortemporal-lh':        12, 19, 25, 27, 28, 36
    # 'superiortemporal-rh':        58, 61, 65, 67, 71, 77
    # 'rostralmiddlefrontal-lh':    30, 32, 35, 39, 41
    # 'rostralmiddlefrontal-rh':    79, 80, 81, 82, 83

    # 'parstriangularis-lh':        75, 76, 79, 83
    # 'parstriangularis-rh':        0, 5

    base_patch = np.array([19, 28, 58, 12, 36, 61, 65, 71, 77, 30, 39, 41, 80, 82, 0, 5, 75, 83, 32, 79])
    aud_patch = np.array([19, 28, 58, 12, 36, 61, 65, 71, 77, 30, 39, 41, 80, 82, 0, 5, 75, 83, 32, 79])
    # base_patch_ = np.array([39, 0, 30, 12, 28, 5, 61, 41, 58, 80, 83, 75])
    corr_thr = 1
    var_thr = 0.9

    # m_active_vec = [2, 3, 4, 5]
    k = 2
    m_active_vec = [k]
    # m_inactive_vec = [0, 2, 4, 6]
    m_inactive_vec = [20-k]
    alpha = -1

    # [58 61 30 77 79 32 83 39 28 12 65  5] = 47, seed(3)
    # [39  0 30 12 28  5 61 41 58 80 83 75] = 46, seed(0)

    tabel_h = np.zeros((len(m_active_vec), len(m_inactive_vec)))
    tabel_f = np.zeros((len(m_active_vec), len(m_inactive_vec)))
    # total_trial += 1
    for m2, m_inactive in enumerate(m_inactive_vec):
        for m1, m_active in enumerate(m_active_vec):
            m = m_active + m_inactive
            msd_det = np.zeros((total_trial, 1))
            fls_det = np.zeros((total_trial, 1))
            J_all = np.zeros((total_trial, m, m))
            patch_all = []
            # for k in range(0, total_trial):
            k = 0
            while k < total_trial:
                try:
                    cnt = 0
                    corr_penalty = 0
                    while True:
                        # patch_idx = base_patch[np.random.permutation(range(len(base_patch)))][:m_active+m_inactive]
                        act_patch = aud_patch[np.random.permutation(range(len(aud_patch)))][:m_active]
                        inact_patch = list(range(m_inactive_vec[0]))
                        # for w in act_patch:
                        #     inact_patch.remove(w)
                        patch_idx = np.hstack((act_patch, np.array(inact_patch)))
                        ex_g, g, x, y, r_cov, p, JG = data_generation(patch_idx, m_active, alpha, evoked, forward, cov, labels_as_list, n_eigenmodes)
                        corr = np.abs(np.corrcoef(ex_g.T))
                        np.fill_diagonal(corr, 0)
                        cnt += 1
                        if cnt > 5:
                            corr_penalty += 1
                        if np.sum(corr > corr_thr) <= corr_penalty:
                            break
                    print('total patch:', len(patch_idx))
                    print('active patch: ', act_patch)
                    print('trial: ', k)
                    t, n = y.shape
                    f = ex_g
                    ROIs = list(range(0, m))

                    d = np.zeros((m, m))
                    d_b = np.zeros((m, m))
                    tt = t // n_segments
                    for n_ in range(0, n_segments):
                        print('segment: ', n_)
                        dev_raw, bias_r, bias_f, model_f, conv_flag = \
                                _gc_extraction(y[n_ * tt: (n_ + 1) * tt, :].T, f, r=r_cov * np.eye(n), p=p, p1=p,
                                               n_eigenmodes=n_eigenmodes, ROIs=ROIs, lambda_range=lambda_range,
                                               max_iter=max_iter, max_cyclic_iter=max_cyclic_iter,
                                               tol=tol, sparsity_factor=sparsity_factor, var_thr=var_thr)

                        dev_raw[dev_raw < -10] = 0
                        d_b += dev_raw
                        d += debiased_dev(dev_raw, bias_f, bias_r)
                        # print(debiased_dev(dev_raw, bias_f, bias_r))

                    d_b /= n_segments
                    d /= n_segments
                    J = fdr_control(d, p*n_eigenmodes, 0.001)

                    msd_det[k], fls_det[k] = missed_false_detection(JG, J)
                    print('missed: ', msd_det[k] / np.sum(JG))
                    print('false : ', fls_det[k] / (m * (m - 1) - np.sum(JG)))
                    np.set_printoptions(precision=2)
                    # print('avg_dev: \n', d[:m_active, :m_active])
                    patch_all.append(patch_idx)
                    J_all[k] = J
                    k += 1
                except ValueError:
                    print('ValueError! Run it again!')
                    continue
                except np.linalg.LinAlgError:
                    print('LinAlgError! Run it again!')
                    continue
                except RuntimeError:
                    print('RuntimeError! Run it again!')
                    continue

            print('Summary: ')
            print('# of actives:', m_active)
            print('# of inactives:', m_inactive)
            print('Total sources:', m)
            print('Alpha:', alpha)
            print('hit rate: ', 1 - msd_det.mean()/np.sum(JG), ' (', (1 - msd_det/np.sum(JG)).std(), ')')
            print('false alarm : ', fls_det.mean()/(m*(m-1)-np.sum(JG)), ' (', fls_det.std()/(m*(m-1)-np.sum(JG)), ')')
            print('------------------------------------------------------------------')
            tabel_h[m1, m2] = 1 - msd_det.mean()/np.sum(JG)
            tabel_f[m1, m2] = fls_det.mean()/(m*(m-1) - np.sum(JG))
            obj = DiagFls(m_active, m_inactive, JG, J_all, patch_all)

            # file_name = f"{m_active}-{m_inactive}-{alpha}.pkl"
            # with open(file_name, 'wb') as fp:
            #     pickle.dump(obj, fp)

    # file_name = f"{m_active}-{m_inactive}-{alpha}.pkl"
    # with open(file_name, 'rb') as fp: temp = pickle.load(fp)


    # a_f = model_f._parameters[0]
    # q_f = model_f._parameters[2]
    #
    # a_f_ = np.zeros((m, m))
    # for i in range(0, n_eigenmodes):
    #     a_f_ += np.squeeze(np.abs(a_f[0, i::n_eigenmodes, i::n_eigenmodes]))
    #
    # # print(dev_raw)

    # al = model_f.mse_path.mean(axis=1)
    # fig, ax = plt.subplots(6)
    # for a, ll_ in zip(ax, al): a.plot(model_f.cv_lambdas, ll_)
    # for a in ax: a.set_xscale('log')


