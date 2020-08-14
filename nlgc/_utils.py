"Deviance calculation"

import numpy as np

def bias(q, a, x_, idx_src):
    """Computes the bias in the deviance

        Parameters
        ----------
        q:  ndarray of shape (n_sources*mo, n_sources*mo)
        a:  ndarray of shape (n_sources*mo, n_sources*order*mo)
        x_:  ndarray of shape (t, n_sources*mo)
        idx_src: source index

        Returns
        -------
        bias

    """
    t, dxm = x_.shape
    _, dtot = a.shape
    p = int(dtot / dxm)

    ai = a[idx_src, :].T

    xi = x_[p:t, idx_src]

    cx = np.zeros((t-p, dtot))

    for i in range(dxm):
        for k in range(p):
            cx[:, i*p + k] = x_[p-1-k: t-1-k, i]

    qd = np.diag(q)

    # gradient of log - likelihood
    ldot = np.zeros((dtot + 1, 1))

    ldot[0, 0] = 1/qd[idx_src]*(t-p)/2*(1-1/2/np.pi)
    ldot[1:, 0] = 1/qd[idx_src]*cx.T@(xi - cx@ai)

    # hessian of log - likelihood
    ldotdot = np.zeros((dtot + 1, dtot + 1))

    ldotdot[0, 0] = -1/(qd[idx_src] ** 2)*(t - p)/ 2*(1 - 1/2/np.pi)
    ldotdot[1:, 0] = -1 / (qd[idx_src] ** 2)*cx.T@(xi - cx@ai)
    ldotdot[0, 1:] = ldotdot[1:, 0] .T
    ldotdot[1:, 1:] = -1 / qd[idx_src]*(cx.T@cx)

    return ldot.T@np.linalg.solve(ldotdot, ldot)


def test_bias():
    q = np.array([[2, 0], [0, 6]])
    a = np.array([[0.9, 0], [-0.5, 0.3]])
    x_ = np.array([[0.5, 0.5], [-0.5, -0.3], [0.2, 0.7], [-0.1, 0.4]])
    idx_src = 1
    dev_bias = bias(q, a, x_, idx_src)
    print(dev_bias)


def debiased_dev(xr_, xf_, ar, af, qr, qf, t, idx_reg, mo=1):
    """Computes debiased deviance

        Parameters
        ----------
        xr_: (reduced model) ndarray of shape (t, n_sources*order*mo)
        xf_: (full model) ndarray of shape (t, n_sources*order*mo)
        ar:  (reduced model) ndarray of shape (n_sources*mo, n_sources*order*mo)
        af:  (full model) ndarray of shape (n_sources*mo, n_sources*order*mo)
        qr:  (reduced model) ndarray of shape (n_sources*mo, n_sources*mo)
        qf:  (full model) ndarray of shape (n_sources*mo, n_sources*mo)
        t:   number of time samples
        idx_reg: region index
        mo:   number of eigen modes

        Returns
        -------
        deviance : debiased deviance

    """

    deviance = 0
    for idx_src in range(idx_reg * mo, (1 + idx_reg) * mo):
        deviance += t*np.log(qr[idx_src, idx_src] / qf[idx_src, idx_src])
        deviance -= bias(qf, af, xf_, idx_src) - bias(qr, ar, xr_, idx_src)

    return deviance


def test_debiased_dev():
    n = 2
    mo = 2
    p = 1
    t = 10
    qf = np.diag(np.random.uniform(1, 10, n*mo))
    qr = qf + np.diag(np.random.uniform(0, 2, n*mo))

    af = np.random.uniform(0, 1, (n*mo, n*mo*p))
    ar = np.random.uniform(0, 1, (n*mo, n*mo*p))

    x_ = np.random.normal(0, 1, (t, n*mo))

    for idx_reg in range(n):
        dev_bias = debiased_dev(x_, x_, ar, af, qr, qf, t, idx_reg, mo)
        print(dev_bias)
