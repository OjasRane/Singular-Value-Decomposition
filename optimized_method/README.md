# Optimized Method

This directory contains an optimized implementation of Singular Value Decomposition (SVD) using NumPy's built-in linear algebra routines.

## Contents

- `optimized_svd.py`: Provides functions for computing and reconstructing SVD in a compact way.

## Implementation

- `optimized_svd(A)`: Computes the singular value decomposition of matrix `A`.
  - Returns `U`, `S`, and `Vt` using `numpy.linalg.svd` with `full_matrices=False`.
- `reconstruct(U, S, Vt, k=None)`: Reconstructs the original matrix from its SVD factors.
  - If `k` is provided, it performs a rank-`k` approximation by truncating the SVD components.
  - Uses `_extract_diagonal` from `metrics.metrics` to ensure singular values are handled correctly.

## Dependencies

- `numpy`
- `metrics/metrics.py` from the project for `_extract_diagonal`

## Example

```python
import numpy as np
from optimized_method.optimized_svd import optimized_svd, reconstruct

A = np.random.rand(100, 80)
U, S, Vt = optimized_svd(A)
A_approx = reconstruct(U, S, Vt, k=10)
```

## Purpose

This module is intended to provide a practical SVD path in this project by leveraging NumPy's optimized implementation, while still supporting low-rank reconstruction for compression and approximation experiments.
