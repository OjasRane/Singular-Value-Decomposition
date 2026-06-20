import numpy as np

def extract_diagonal(S):
    if S.ndim == 2:
        S = np.diag(S)
    return S