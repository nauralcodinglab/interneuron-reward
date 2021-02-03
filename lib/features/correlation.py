import numpy as np
from scipy import stats

def vectorized_spearman_corr(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Get the correlation between vec and each row of a matrix.

    Returns
    -------
    np.ndarray
        Spearman non-parametric correlation coefficient between vec and each
        row of mat.
    """
    assert mat.ndim == 2
    assert vec.ndim == 1
    assert mat.shape[1] == len(vec)

    return np.apply_along_axis(lambda tr: stats.spearmanr(tr, vec)[0], 1, mat)
