import numpy as np
from metrics.metrics import _extract_diagonal

def optimized_svd(A, dtype=np.float64):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return U.astype(dtype), S.astype(dtype), Vt.astype(dtype)

def reconstruct(U, S, Vt, k=None):
    if k is not None:
        U = U[:, :k]
        S = _extract_diagonal(S)[:k]
        Vt = Vt[:k]

    Sigma = np.diag(S)
    return U @ Sigma @ Vt