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
        W = np.hstack((np.random.uniform(0.75, 1.25, (1, n_eigenmodes)),
                       np.random.uniform(0, np.sqrt(2)/2, (1, n2-n_eigenmodes))))
        W *= np.sign(np.random.uniform(-1, 1, (1, n2)))
    else:
        n2 = 3
        W = np.array([1, 1, alpha])

    G, _ = nlgc_map_(evoked, forward, cov, labels_as_list, n_eigenmodes=n2)

    n, _ = G.shape
    g = np.zeros((n, len(patch_idx)))
    ex_g = np.zeros((n, n_eigenmodes * len(patch_idx)))

    for idx, this_patch in enumerate(patch_idx):
        ex_g[:, idx * n_eigenmodes: (idx + 1) * n_eigenmodes] = G[:, this_patch * n2: this_patch * n2 + n_eigenmodes]
        g_ = G[:, this_patch * n2: (this_patch + 1) * n2]

        g0 = np.sum(g_ * W, axis=1)
        g0 /= np.sum(W)
        g[:, idx] = g0 / np.linalg.norm(g0, ord=2)

    ex_g /= np.sqrt(np.sum(ex_g ** 2, axis=0))

    # Simulate AR process
    ###############################################################################
    m, p, t = len(patch_idx), 2, 3000

    q = 0.1*m_active/(m-m_active)*np.eye(m)

    a = np.zeros(p * m * m, dtype=np.float64)
    a.shape = (p, m, m)

    for i in range(m_active):
        q[i, i] = 1
        a[1, i, i] = 0.9

    if m_active >= 2:
        a[0, 0, 1] = -0.5

    if m_active >= 3:
        a[0, 0, 2] = -0.5

    if m_active >= 4:
        a[0, 3, 2] = -0.5

    if m_active >= 5:
        a[0, 3, 4] = -0.5

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


def return_neighbors(src_target, idx):
    hm = 0
    if idx >= 42:
        idx -= 42
        hm = 1
    vertno = src_target[hm]['vertno'][idx]

    neighbors_vertno = np.argsort(src_target[hm]['dist'][vertno].toarray())[0][:-42:-1][::-1][:6]
    neighbors_ = []
    for n_v in neighbors_vertno:
        for i in range(42):
            if n_v in src_target[hm]['pinfo'][i]:
                neighbors_.append(i)

    if hm == 1:
        neighbors = [n + 42 for n in neighbors_]
        neighbors.append(42+idx)
    else:
        neighbors = neighbors_
        neighbors.append(idx)

    return neighbors


def relaxed_rates(src_target, JG, J, patch_idx):
    nn_true_links = []
    for tar, src in zip(np.nonzero(JG)[0], np.nonzero(JG)[1]):
        nn_true_links.append((return_neighbors(src_target, patch_idx[src]), return_neighbors(src_target, patch_idx[tar])))

    hit_rate = 0
    false_rate = 0
    disc_links = []
    for tar, src in zip(np.nonzero(J)[0], np.nonzero(J)[1]):
            disc_links.append((patch_idx[src], patch_idx[tar]))

    for link in disc_links:
        flag = 0
        for i in range(len(nn_true_links)):
            if link[0] in nn_true_links[i][0] and link[1] in nn_true_links[i][1]:
                hit_rate += 1
                flag = 1
                break
        if flag == 0:
            false_rate += 1

    m, _ = JG.shape
    return np.min([1, hit_rate/np.sum(JG)]), false_rate/(m*(m-1) - np.sum(JG))


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

    total_trial = 10
    n_eigenmodes = 2
    lambda_range = np.asanyarray([1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
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

    aud_patch = np.array([19, 28, 58, 12, 36, 61, 65, 71, 77, 30, 39, 41, 80, 82, 0, 5, 75, 83, 32, 79])
    corr_thr = 0.65
    var_thr = 0.95

    m_active_vec = [2]
    m_inactive_vec = [2]
    alpha = -1

    tabel_h = np.zeros((len(m_active_vec), len(m_inactive_vec)))
    tabel_f = np.zeros((len(m_active_vec), len(m_inactive_vec)))
    tabel_rlx_h = np.zeros((len(m_active_vec), len(m_inactive_vec)))
    tabel_rlx_f = np.zeros((len(m_active_vec), len(m_inactive_vec)))
    for m2, m_inactive in enumerate(m_inactive_vec):
        for m1, m_active in enumerate(m_active_vec):
            m = m_active + m_inactive
            msd_det = np.zeros((total_trial, 1))
            fls_det = np.zeros((total_trial, 1))
            relaxed_hit_rate = np.zeros((total_trial, 1))
            relaxed_fls_rate = np.zeros((total_trial, 1))
            J_all = np.zeros((total_trial, m, m))
            patch_all = []
            k = 0
            while k < total_trial:
                try:
                    cnt = 0
                    corr_penalty = 0
                    while True:
                        act_patch = aud_patch[np.random.permutation(range(len(aud_patch)))][:m_active]
                        inact_patch_ = list(range(84))
                        for w in act_patch:
                            inact_patch_.remove(w)
                        inact_patch_ = np.array(inact_patch_)
                        inact_patch = inact_patch_[np.random.permutation(range(len(inact_patch_)))][:m_inactive]
                        patch_idx = np.hstack((act_patch, np.array(inact_patch)))
                        ex_g, g, x, y, r_cov, p, JG = data_generation(patch_idx, m_active, alpha, evoked, forward, cov, labels_as_list, n_eigenmodes)
                        corr = np.abs(np.corrcoef(ex_g[:, :m_active*n_eigenmodes].T))
                        np.fill_diagonal(corr, 0)
                        cnt += 1
                        if cnt > 50:
                            corr_penalty += 1
                        if np.sum(corr > corr_thr) <= corr_penalty:
                            break
                    print('total patch:', len(patch_idx))
                    print('active patch: ', act_patch)
                    print('inactive patch: ', inact_patch)
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

                    d_b /= n_segments
                    d /= n_segments
                    J = fdr_control(d, p*n_eigenmodes, 0.001)

                    msd_det[k], fls_det[k] = missed_false_detection(JG, J)
                    relaxed_hit_rate[k], relaxed_fls_rate[k] = relaxed_rates(src_target, JG, J, patch_idx)
                    print('################################################################')
                    print('missed: ', msd_det[k] / np.sum(JG))
                    print('false : ', fls_det[k] / (m * (m - 1) - np.sum(JG)))
                    print('*****************************************************************')
                    print('relaxed missed: ', 1-relaxed_hit_rate[k])
                    print('relaxed false : ', relaxed_fls_rate[k])
                    print('################################################################')
                    np.set_printoptions(precision=2)
                    patch_all.append(patch_idx)
                    J_all[k] = J
                    k += 1
                except ValueError:
                    print('ValueError! Run it again!')
                    continue
                except np.linalg.LinAlgError:
                    print('LinAlgError! Run it again!')
                    continue
                except RuntimeError as err:
                    print('handling RuntimeError!', err)
                    continue

            print('Summary: ')
            print('# of actives:', m_active)
            print('# of inactives:', m_inactive)
            print('Total sources:', m)
            print('Alpha:', alpha)
            print('hit rate: ', 1 - msd_det.mean()/np.sum(JG), ' (', (1 - msd_det/np.sum(JG)).std(), ')')
            print('false alarm : ', fls_det.mean()/(m*(m-1)-np.sum(JG)), ' (', fls_det.std()/(m*(m-1)-np.sum(JG)), ')')
            print('*****************************************************************')
            print('relaxed hit rate: ', relaxed_hit_rate.mean(), ' (', relaxed_hit_rate.std(), ')')
            print('relaxed false alarm : ', relaxed_fls_rate.mean(), ' (', relaxed_fls_rate.std(), ')')
            print('------------------------------------------------------------------')
            tabel_h[m1, m2] = 1 - msd_det.mean()/np.sum(JG)
            tabel_f[m1, m2] = fls_det.mean()/(m*(m-1) - np.sum(JG))
            tabel_rlx_h[m1, m2] = relaxed_hit_rate.mean()
            tabel_rlx_f[m1, m2] = relaxed_fls_rate.mean()

            # # save all trials to check the false discoveries
            # obj = DiagFls(m_active, m_inactive, JG, J_all, patch_all)
            # file_name = f"{m_active}-{m_inactive}-{alpha}.pkl"
            # with open(file_name, 'wb') as fp:
            #     pickle.dump(obj, fp)

    # # open the false discoveries
    # file_name = f"{m_active}-{m_inactive}-{alpha}.pkl"
    # with open(file_name, 'rb') as fp: temp = pickle.load(fp)



