import numpy as np

def optimized_svd(A):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return U, S, Vt

def reconstruct(U, S, Vt, k=None):
    if k is not None:
        U = U[:, :k]
        S = S[:k]
        Vt = Vt[:k]

    Sigma = np.diag(S)
    return U @ Sigma @ Vt