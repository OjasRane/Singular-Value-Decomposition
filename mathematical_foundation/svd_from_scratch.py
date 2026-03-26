import numpy as np

def svd_from_scratch(A):
    m, n = A.shape
    AAt = A @ A.T
    eigenvalues, eigenvectors = np.linalg.eigh(AAt)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    U = eigenvectors
    Vt = np.zeros((n, n))
    singular_values = np.sqrt(np.clip(eigenvalues, 0, None))
    S = np.zeros((m, n))
    for i in range(min(m, n)):
        S[i, i] = singular_values[i]
    for i in range(min(m, n)):
        if singular_values[i] > 1e-15:
            Vt[i, :] = (A.T @ U[:, i]) / singular_values[i]
    return U, S, Vt