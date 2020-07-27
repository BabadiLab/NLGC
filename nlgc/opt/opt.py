import logging
import os
from multiprocessing import shared_memory, current_process

import numpy as np
from joblib import Parallel, delayed
from sklearn import preprocessing
from sklearn.model_selection import TimeSeriesSplit

from nlgc.opt.e_step import sskf, align_cast
from nlgc.opt.m_step import calculate_ss, solve_for_a, solve_for_q, compute_ll

filename = os.path.realpath(os.path.join(__file__, '..', '..', "debug.log"))
logging.basicConfig(filename=filename, level=logging.DEBUG)


class NeuraLVAR:
    """Neural Latent Vector Auto-Regressive model

    Provides a vector auto-regressive model for the unobserved (latent) source activities
    that gives rise to m/eeg data.

    Parameters
    ----------
    order : int
    copy : bool, default=True
    standardize : bool, default=False
    normalize : bool, default=False
    use_lapack : bool, default=True
    """
    _dims = None
    _parameters = None
    _lls = None
    lambda_ = None

    def __init__(self, order, copy=True, standardize=False, normalize=False, use_lapack=True):
        if standardize is not False and normalize is not False:
            raise ValueError(f"both standardize={standardize} and normalize={normalize} cannot be specified")
        elif standardize:
            _preprocessing = preprocessing.StandardScaler(copy)
        elif normalize is not False:
            if isinstance(normalize, bool):
                normalize = 'l2'
            _preprocessing = preprocessing.Normalizer(normalize, copy)
        else:
            _preprocessing = None
        self._preprocessing = _preprocessing
        self._copy = copy
        self._order = order
        self._use_lapack = use_lapack

    def _fit(self, y, f, r, lambda2=None, max_iter=20, max_cyclic_iter=2, a_init=None, q_init=None,
             rel_tol=0.0001, xs=None):
        """Internal function that fits the model from given data

        Parameters
        ----------
        y : ndarray of shape (n_channels, n_samples)
        f : ndarray of shape (n_channels, n_sources)
        r : ndarray of shape (n_channels, n_channels)
        lambda2 : float, default=None
        max_iter : int, default=100
        max_cyclic_iter : int, default=2
        a_init : ndarray of shape (order, n_sources, n_sources), default=None
        q_init : ndarray of shape (n_sources, n_sources), default=None
        rel_tol : float, default=0.0001
        xs : tuple of two ndarrays of shape (n_samples, n_sources), default=None

        Returns
        -------
        a : ndarray of shape (order, n_sources, n_sources)
        q : ndarray of shape (n_sources, n_sources),
        lls : list
            list containing the likelihood values along the training path
        f : ndarray of shape (n_channels, n_sources)
        r : ndarray of shape (n_channels, n_channels)
        xs : tuple of two ndarrays of shape (n_samples, n_sources)
            Used for mostly reusing the allocated memories
        """
        y, a_, a_upper, f_, q_, q_upper, non_zero_indices, r, xs, m, n, p, use_lapack = \
            self._prep_for_sskf(y, a_init, f, q_init, r, xs)

        lls = []
        for i in range(max_iter):
            a_[:m] = a_upper
            q_[non_zero_indices] = q_upper[non_zero_indices]

            x_, s_, b, s_hat = sskf(y, a_, f_, q_, r, xs=xs, use_lapack=use_lapack)
            ll = compute_ll(y, x_, s_, s_hat, a_upper, f, q_upper, r, m, n, p)
            lls.append(ll)
            # stopping cond
            if i > 0:
                rel_change = (lls[i - 1] - lls[i]) / lls[i - 1]
                if rel_change < rel_tol:
                    break

            s1, s2, s3 = calculate_ss(x_, s_, b, m, p)

            for _ in range(max_cyclic_iter):
                a_upper, changes = solve_for_a(q_upper, s1, s2, a_upper, lambda2=lambda2, max_iter=1000, tol=0.01)
                q_upper = solve_for_q(q_upper, s3, s1, a_upper, lambda2=lambda2)

        a = self._unravel_a(a_upper)
        return a, q_upper, lls, f, r, xs

    def compute_ll(self, y, args=None):
        """Returns log(p(y|args=(a, f, q, r))).

        Parameters
        ----------
        y : ndarray of shape (n_channels, n_samples)
        args : tuple of ndarrays, default=None
            Expect a tuple with the model parameters: (a, f, q, r).
            if None, self._parameters is used.

        Returns
        -------
        log_likelihood : float
            Returns log(p(y|(a, f, q, r))).
        """
        if args is None:
            args = self._parameters
        a, f, q, r = args
        y, a_, a_upper, f_, q_, q_upper, _, r, (_x, x_), m, n, p, use_lapack = self._prep_for_sskf(y, a, f, q, r)
        x_, s_, b, s_hat = sskf(y, a_, f_, q_, r, xs=(_x, x_), use_lapack=use_lapack)
        ll = compute_ll(y, x_, s_, s_hat, a_upper, f, q_upper, r, m, n, p)
        return ll

    def fit(self, y, f, r, lambda2=None, max_iter=100, max_cyclic_iter=2, a_init=None, q_init=None, rel_tol=0.0001):
        """Fits the model from given m/eeg data, forward gain and noise covariance

        Parameters
        ----------
        y : ndarray of shape (n_channels, n_samples)
        f : ndarray of shape (n_channels, n_sources)
        r : ndarray of shape (n_channels, n_channels)
        lambda2 : float, default=None
        max_iter : int, default=100
        max_cyclic_iter : int, default=2
        a_init : ndarray of shape (order, n_sources, n_sources), default=None
        q_init : ndarray of shape (n_sources, n_sources), default=None
        rel_tol : float, default=0.0001
        """
        a, q_upper, lls, f, r, _ = self._fit(y, f, r, lambda2=lambda2, max_iter=max_iter,
                                             max_cyclic_iter=max_cyclic_iter,
                                             a_init=a_init, q_init=q_init, rel_tol=rel_tol)
        self._parameters = (a, f, q_upper, r)
        self._lls = lls
        self.lambda_ = lambda2
        return self

    @staticmethod
    def _ravel_a(a):
        p, m, m_ = a.shape
        assert m == m_
        return np.reshape(np.swapaxes(a, 0, 1), (m, m * p))

    @staticmethod
    def _unravel_a(a):
        m, mp = a.shape
        p = mp // m
        return np.swapaxes(np.reshape(a, (m, p, m)), 0, 1)

    def _prep_for_sskf(self, y, a, f, q, r, xs=None):
        """Prepares (mostly memory allocation and type-casting) arrays for sskf()

        Parameters
        ----------
        y : ndarray of shape (n_channels, n_samples)
        a : ndarray of shape (order, n_sources, n_sources), None
        f : ndarray of shape (n_channels, n_sources)
        q : ndarray of shape (n_sources, n_sources)
        r : ndarray of shape (n_channels, n_channels)
        xs : tuple of two ndarrays of shape (n_sources*order, n_samples), default=None

        Returns
        -------
        y : ndarray of shape (n_samples, n_channels)
        a_ : ndarray of shape (n_sources*order, n_sources*order)
        a_upper : ndarray of shape (n_sources, n_sources*order)
        f_ : ndarray of shape (n_channels, n_channels*order)
        q_ : ndarray of shape (n_sources*order, n_sources*order)
        q_upper : ndarray of shape (n_sources, n_sources)
        non_zero_indices : tuple of ndarrays of shape (n_sources, )
            (non-zero) diag indices of (q_) q_upper
        r : ndarray of shape (n_samples, samples)
        xs : tuple of two ndarrays of shape (n_sources*order, n_samples)
        m : int
            n_sources
        n : int
            n_channles
        p : int
            order
        use_lapack: bool
        """
        n, m = f.shape
        _, t = y.shape
        p = self._order
        use_lapack = self._use_lapack

        if y.flags['F_CONTIGUOUS']:
            y = y.T
        else:
            y = y.copy(order='F').T  # to make y contiguous in 'F'

        assert q.shape[0] == m
        q_upper = q

        a_upper = np.zeros((m, m * p), dtype=np.float64)
        a_lower = np.hstack((np.eye(m * (p - 1)), np.zeros((m * (p - 1), m))))
        a_ = np.vstack((a_upper, a_lower))
        if a is not None:
            a_upper[:] = self._ravel_a(a)
            a_[:m] = a_upper

        q_ = np.zeros((m * p, m * p))
        non_zero_indices = np.diag_indices_from(q_upper)
        q_[non_zero_indices] = q_upper[non_zero_indices]

        f_ = np.hstack((f, np.zeros((n, m * (p - 1)))))

        if xs is None:
            _x = np.empty((t, m * p), dtype=np.float64)
            x_ = np.empty_like(_x)
            xs = (_x, x_)
        else:
            _x, x_ = xs
            assert _x.shape[0] == y.shape[0]
            assert _x.flags['C_CONTIGUOUS']
            assert x_.flags['C_CONTIGUOUS']
        return y, a_, a_upper, f_, q_, q_upper, non_zero_indices, r, xs, m, n, p, use_lapack


class NeuraLVARCV(NeuraLVAR):
    """Neural Latent Vector Auto-Regressive model (supports cross-validation)

    Provides a vector auto-regressive model for the unobserved (latent) source activities
    that gives rise to m/eeg data.

    Parameters
    ----------
    order : int
        VAR order
    max_n_mus : int
        keep it 5
    cv : int | sklearn.model_selection split objects
    n_jobs : int
    copy : bool, default=True
    standardize : bool, default=False
    normalize : bool, default=False
    use_lapack : bool, default=True
    """
    cv_lambdas = None
    mse_path = None

    def __init__(self, order, max_n_mus, cv, n_jobs, copy=True, standardize=False, normalize=False, use_lapack=True):
        self.max_n_mus = max_n_mus
        self.cv = cv
        self.n_jobs = n_jobs
        NeuraLVAR.__init__(self, order, copy, standardize, normalize, use_lapack)

    def _cvfit(self, split, info_y, info_f, info_r, info_cv, splits, lambda_range, max_iter=100, max_cyclic_iter=2,
               a_init=None, q_init=None, rel_tol=0.0001):
        """Utility function to be used by self.fit()

        Parameters
        ----------
        split :
        info_y :
        info_f :
        info_r :
        info_cv :
        splits :
        lambda_range :
        max_iter :
        max_cyclic_iter :
        a_init :
        q_init :
        rel_tol :

        Returns
        -------
        val

        """
        logger = logging.getLogger(__name__)
        logger.debug(f"{current_process().name} working on {split}th split")
        try:
            y, shm_y = link_share_memory(info_y)
            f, shm_f = link_share_memory(info_f)
            r, shm_r = link_share_memory(info_r)
            cv, shm_c = link_share_memory(info_cv)
        except BaseException as e:
            logger.error("Could not link to memory")
            raise e

        logger.debug(f"{current_process().name} successfully read the shared memory")
        train, test = splits[split]
        y_train, y_test = y[:, train], y[:, test]
        xs = None
        logger.debug(f"{current_process().name} successfully split the data")
        val = 0
        for i, lambda2 in enumerate(lambda_range):
            logger.debug(f"{current_process().name} {split} doing {lambda2}")
            a_, q_upper, lls, *rest, xs = \
                self._fit(y_train, f, r, lambda2=lambda2, max_iter=max_iter, max_cyclic_iter=max_cyclic_iter,
                          a_init=a_init, q_init=q_init, rel_tol=rel_tol, xs=xs)
            cross_ll = self.compute_ll(y_test, (a_, f, q_upper, r))
            cv[split, i] = cross_ll
            val += cross_ll

        for shm in (shm_y, shm_f, shm_r, shm_c):
            shm.close()
        return val

    def fit(self, y, f, r, lambda_range=None, max_iter=100, max_cyclic_iter=2, a_init=None, q_init=None,
            rel_tol=0.0001):
        """Fits the model from given m/eeg data, forward gain and noise covariance

        y : ndarray of shape (n_channels, n_samples)
        f : ndarray of shape (n_channels, n_sources)
        r : ndarray of shape (n_channels, n_channels)
        lambda_range : ndarray of floats, default=None
        max_iter : int, default=100
        max_cyclic_iter : int, default=2
        a_init : ndarray of shape (order, n_sources, n_sources), default=None
        q_init : ndarray of shape (n_sources, n_sources), default=None
        rel_tol : float, default=0.0001
        """
        y, f = align_cast((y, f), self._use_lapack)  # to make y, f contiguous in 'F'

        if lambda_range is None or lambda_range == 'auto':
            raise NotImplementedError("Try specifying a pre-determined range")

        # do cvsplits
        if isinstance(self.cv, int):
            kf = TimeSeriesSplit(n_splits=self.cv)
        else:
            kf = self.cv
        cvsplits = [split for split in kf.split(y.T)]

        cv_mat = np.zeros((len(cvsplits), len(lambda_range)), dtype=y.dtype)
        # Use parallel processing
        # A, b, mu_range, cv_mat needs to shared across processes
        shared_y, info_y, shm_y = create_shared_mem(y)
        shared_f, info_f, shm_f = create_shared_mem(f)
        shared_r, info_r, shm_r = create_shared_mem(r)
        shared_cv_mat, info_cv, shm_c = create_shared_mem(cv_mat)
        initargs = (info_y, info_f, info_r, info_cv, cvsplits, lambda_range,
                    max_iter, max_cyclic_iter, a_init, q_init, rel_tol)

        Parallel(n_jobs=self.n_jobs, )(delayed(self._cvfit)(i, *initargs) for i in range(len(cvsplits)))

        self.cv_lambdas = lambda_range
        cv_mat[:, :] = np.reshape(shared_cv_mat, cv_mat.shape)
        self.mse_path = cv_mat
        for shm in (shm_y, shm_f, shm_r):
            shm.close()
            shm.unlink()

        # Find best mu
        normalized_cross_lls = self.mse_path - self.mse_path.max()
        index = np.argmax(np.sum(np.exp(normalized_cross_lls), axis=0))
        best_lambda = lambda_range[index]

        a, q_upper, lls, f, r, _ = self._fit(y, f, r, lambda2=best_lambda, max_iter=max_iter,
                                             max_cyclic_iter=max_cyclic_iter,
                                             a_init=a_init, q_init=q_init, rel_tol=rel_tol)
        self._parameters = (a, f, q_upper, r)
        self._lls = lls
        self.lambda_ = best_lambda


def create_shared_mem(arr):
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    info = (arr.shape, arr.dtype, shm.name)
    shared_arr = np.ndarray((arr.size,), dtype=arr.dtype, buffer=shm.buf)
    shared_arr[:] = arr.ravel()[:]
    return shared_arr, info, shm


def link_share_memory(info):
    shape, dtype, name = info
    shm = shared_memory.SharedMemory(name=name)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return arr, shm


def initialize_q(y, f, r):
    q = np.eye(f.shape[1], dtype=np.float64)
    return q