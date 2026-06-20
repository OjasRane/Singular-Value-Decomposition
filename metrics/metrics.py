import numpy as np
from utils import extract_diagonal

def frobenius_error(A, Ak):
    return np.linalg.norm(A-Ak, "fro")

def frobenius_error_squared(A, Ak):
    return np.linalg.norm(A-Ak, "fro")**2

def relative_frobenius_error(A, Ak):
    return np.linalg.norm(A-Ak, "fro") / np.linalg.norm(A, "fro")

def reconstruction_error(S, k):
    S = extract_diagonal(S)
    return np.sqrt(np.sum(S[k:]**2))

def reconstruction_error_squared(S, k):
    S = extract_diagonal(S)
    return np.sum(S[k:]**2)

def energy_retained(S, k):
    S = extract_diagonal(S)
    return np.sum(S[:k]**2) / np.sum(S**2)

def compression_ratio(image_shape, k, dtype=None):
    return k*(np.sum(image_shape[:2]) + 1) / np.prod(image_shape[:2]) if dtype is None else (k*(np.sum(image_shape[:2]) + 1) / np.prod(image_shape[:2])) * np.dtype(dtype).itemsize

def percent_compression_ratio(image_shape, k, dtype=None):
    return 100*k*(np.sum(image_shape[:2]) + 1) / np.prod(image_shape[:2]) if dtype is None else 100*k*(np.sum(image_shape[:2]) + 1) / np.prod(image_shape[:2]) * np.dtype(dtype).itemsize