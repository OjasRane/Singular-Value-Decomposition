import numpy as np

def frobenius_error(A, Ak):
    return np.linalg.norm(A-Ak, "fro")

def frobenius_error_squared(A, Ak):
    return np.linalg.norm(A-Ak, "fro")**2

def relative_frobenius_error(A, Ak):
    return np.linalg.norm(A-Ak, "fro") / np.linalg.norm(A, "fro")

def _extract_diagonal(S):
    if S.ndim == 2:
        S = np.diag(S)
    return S

def reconstruction_error(S, k):
    S = _extract_diagonal(S)
    return np.sqrt(np.sum(S[k:]**2))

def energy_retained(S, k):
    S = _extract_diagonal(S)
    return np.sum(S[:k]**2) / np.sum(S**2)