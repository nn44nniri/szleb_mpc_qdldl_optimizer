# szleb_mpcrl/solvers/csc_utils.py
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def to_upper_csc(M: sp.csc_matrix) -> sp.csc_matrix:
    """
    Keep only upper triangle (including diagonal) in CSC.
    QDLDL expects upper-tri data only. :contentReference[oaicite:8]{index=8}
    """
    M = M.tocsc()
    U = sp.triu(M, k=0).tocsc()
    U.eliminate_zeros()
    return U


def csc_matvec(A: sp.csc_matrix, x: np.ndarray) -> np.ndarray:
    return A.dot(x)


def clip_box(v: np.ndarray, l: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(v, l), u)