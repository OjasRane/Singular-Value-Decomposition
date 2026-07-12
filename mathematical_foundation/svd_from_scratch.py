import numpy as np

def svd_from_scratch(A):
    m, n = A.shape
    max_element = float(np.max(A))
    A = A.astype(np.float64) / max_element
    AtA = A.T @ A
    eigenvalues, eigenvectors = np.linalg.eigh(AtA)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    V = eigenvectors
    singular_values = np.sqrt(np.clip(eigenvalues, 0, None))
    S = np.zeros_like(A, dtype=np.float64)
    for i in range(min(m, n)):
        S[i, i] = singular_values[i]

    U = np.zeros((m, m))
    for i in range(m):
        if singular_values[i] > 1e-10:
            U[:, i] = (A @ np.array(V[:, i]).T) / singular_values[i]

    return U, S*max_element, V.T