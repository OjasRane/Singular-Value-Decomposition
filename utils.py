import numpy as np

def extract_diagonal(S):
    if S.ndim == 2:
        S = np.diag(S)
    return S

def read_data(f, size):
    data = f.read(size)
    if len(data) != size:
        raise ValueError("Corrupted SVD file")
    return data