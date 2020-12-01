# Author: Behrad Soleimani <behrad@umd.edu>

"Statistical tests"

import numpy as np
from scipy.stats import chi2, ncx2


def fdr_control(d, k, alpha):
    """CFDR control based on BY procedure

            Parameters
            ----------
            d:  ndarray of shape (n_sources, n_sources)
                deviance matrix
            k:  ndarray of shape (n_sources, n_sources)
                degrees of freedom
            alpha:  FDR rate

            Returns
            -------
            j_val: ndarray of shape (n_sources, n_sources)

        """
    if isinstance(k, (list, tuple, np.ndarray)) == 0:
        k = k * np.ones_like(d, dtype=int)

    _, n = d.shape

    i = n * (n - 1)
    alpha_bar = alpha * (i + 1) / (2 * i * np.log(i))

    p_row = 1 - chi2.cdf(d.ravel(), k.ravel())
    sorted_idx = np.argsort(p_row)
    p_val = p_row[sorted_idx]

    # Conversion:  row = np.floor(idx) ; col = idx - row*n
    temp = np.array(range(1, n ** 2 + 1)) * alpha / i / np.log(i)
    temp = np.nonzero(p_val > temp)

    if temp[0].size == 0:
        gc_idx = []
    else:
        reject_idx = temp[0][0]
        gc_idx = sorted_idx[:reject_idx]

    j_val = np.zeros_like(d, dtype=float)
    for i in range(len(gc_idx)):
        row = gc_idx[i] // n
        col = gc_idx[i] % n
        non_centrality = d[row, col] - k[row, col] if d[row, col] > k[row, col] else 0
        j_val[row, col] = 1 - alpha_bar - ncx2.cdf(1 - chi2.isf(1 - alpha_bar, k[row, col]), k[row, col],
                                                   non_centrality)

    return j_val


def test_fdr_control():
    k = np.array([[1, 2], [2, 1]])
    d = np.array([[0, 4], [3, 0]])
    alpha = 0.1
    j_val = fdr_control(d, k, alpha)
    assert np.allclose(j_val, 0)
