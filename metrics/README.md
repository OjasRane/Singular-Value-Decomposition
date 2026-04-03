# Metrics Module

This module provides utility functions for evaluating the quality and efficiency of SVD-based image compression. It offers metrics to measure reconstruction error, energy retention, and compression ratios.

## Overview

The metrics module contains several functions to assess how well the compressed image preserves the original data and how efficiently the compression is achieved. These metrics are essential for tuning the compression parameter `k` (number of singular values retained).

## Functions

### Reconstruction Error Metrics

#### `frobenius_error(A, Ak)`
Computes the Frobenius norm of the difference between original matrix `A` and reconstructed matrix `Ak`.

**Parameters:**
- `A`: Original matrix (numpy array)
- `Ak`: Reconstructed matrix (numpy array)

**Returns:** Float representing the absolute reconstruction error

**Formula:** $ \|A - A_k\|_F = \sqrt{\sum_{i,j} (A_{ij} - A_{k,ij})^2} $

**Example:**
```python
from metrics.metrics import frobenius_error
import numpy as np

A = np.random.rand(100, 100)
Ak = np.random.rand(100, 100)
error = frobenius_error(A, Ak)
```

---

#### `frobenius_error_squared(A, Ak)`
Computes the squared Frobenius norm. Useful for optimization algorithms.

**Parameters:**
- `A`: Original matrix (numpy array)
- `Ak`: Reconstructed matrix (numpy array)

**Returns:** Float representing the squared reconstruction error

**Formula:** $ \|A - A_k\|_F^2 = \sum_{i,j} (A_{ij} - A_{k,ij})^2 $

---

#### `relative_frobenius_error(A, Ak)`
Normalizes the Frobenius error by the norm of the original matrix, making it scale-invariant.

**Parameters:**
- `A`: Original matrix (numpy array)
- `Ak`: Reconstructed matrix (numpy array)

**Returns:** Float between 0 and 1 representing the relative error (0 = perfect reconstruction)

**Formula:** $ \frac{\|A - A_k\|_F}{\|A\|_F} $

**Example:**
```python
from metrics.metrics import relative_frobenius_error

rel_error = relative_frobenius_error(original_image, compressed_image)
print(f"Relative error: {rel_error:.4f}")  # Low values are better
```

---

### Singular Value-Based Metrics

#### `reconstruction_error(S, k)`
Computes reconstruction error based on discarded singular values.\
This and `frobenius_error` are same just the difference is that `frobenius_error` accepts the inputs as original matrix and truncated matrix whereas `reconstruction_error` takes the input of $\Sigma$ matrix and the number of singular values used.

**Parameters:**
- `S`: Singular values (1D array or 2D diagonal matrix)
- `k`: Number of singular values retained

**Returns:** Float representing the reconstruction error from discarded singular values

**Formula:** $ \sqrt{\sum_{i=k}^{n} \sigma_i^2} $ where $\sigma_i$ are singular values

**Note:** This function automatically extracts singular values from a diagonal matrix if provided.

---

#### `reconstruction_error_squared(S, k)`
Squared version of reconstruction error.

**Parameters:**
- `S`: Singular values (1D array or 2D diagonal matrix)
- `k`: Number of singular values retained

**Returns:** Float representing squared reconstruction error

**Formula:** $ \sum_{i=k}^{n} \sigma_i^2 $

---

#### `energy_retained(S, k)`
Computes the fraction of energy (in terms of singular values) retained after compression.

**Parameters:**
- `S`: Singular values (1D array or 2D diagonal matrix)
- `k`: Number of singular values retained

**Returns:** Float between 0 and 1 representing the fraction of energy retained

**Formula:** $ \frac{\sum_{i=0}^{k-1} \sigma_i^2}{\sum_{i=0}^{n-1} \sigma_i^2} $

**Example:**
```python
from metrics.metrics import energy_retained

# Check that 50 singular values retain 95% of energy
energy = energy_retained(singular_values, 50)
if energy >= 0.95:
    print(f"Excellent compression: {energy*100:.2f}% energy retained")
```

---

### Compression Ratio Metrics

#### `compression_ratio(image_shape, k)`
Computes the compression ratio as the ratio of compressed data size to original data size.

**Parameters:**
- `image_shape`: Shape of the image/matrix (tuple, e.g., (height, width) for grayscale or (height, width, channels) for color)
- `k`: Number of singular values retained

**Returns:** Float representing the compression ratio (< 1 is good compression)

**Formula (for 2D):** $ \frac{k \cdot (m + n + 1)}{m \cdot n} $ where m, n are image dimensions

**Note:** The SVD representation requires storing U (m×k), S (k), and V^T (k×n), hence the k×(m+n+1) numerator.

**Example:**
```python
from metrics.metrics import compression_ratio

ratio = compression_ratio((256, 256), k=50)
print(f"Compression ratio: {ratio:.4f}")  # 0.05 means 95% reduction
```

---

#### `percent_compression_ratio(image_shape, k)`
Computes the compression ratio as a percentage of the original data size.

**Parameters:**
- `image_shape`: Shape of the image/matrix (tuple)
- `k`: Number of singular values retained

**Returns:** Float representing the compression ratio as a percentage (lower is better)

**Formula (for 2D):** $ 100 \cdot \frac{k \cdot (m + n + 1)}{m \cdot n} $

**Example:**
```python
from metrics.metrics import percent_compression_ratio

percent = percent_compression_ratio((256, 256), k=50)
print(f"Compressed to {percent:.2f}% of original size")
```

---

## Usage Examples

### Complete Compression Quality Assessment

```python
import numpy as np
import cv2 as cv
from metrics.metrics import (
    relative_frobenius_error, 
    energy_retained, 
    compression_ratio,
    percent_compression_ratio
)
from optimized_method.optimized_svd import optimized_svd

# Load image
img = cv.imread("image.jpg", cv.IMREAD_GRAYSCALE)
img = img.astype(np.float64)

# Perform SVD
U, S, Vt = optimized_svd(img)

# Evaluate k=50
k = 50
Ak = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

# Calculate metrics
rel_error = relative_frobenius_error(img, Ak)
energy = energy_retained(S, k)
comp_ratio = compression_ratio(img.shape, k)

print(f"Relative error: {rel_error:.4f}")
print(f"Energy retained: {energy*100:.2f}%")
print(f"Compression ratio: {comp_ratio:.4f}")
```

### Finding Optimal k

```python
from metrics.metrics import energy_retained

# Find minimum k that retains 95% energy
target_energy = 0.95
min_k = 1
for k in range(1, len(singular_values)):
    if energy_retained(singular_values, k) >= target_energy:
        min_k = k
        break

print(f"Minimum k to retain {target_energy*100}% energy: {min_k}")
```

## See Also

- [Image Compression Module](../application/README.md) - High-level image compression utilities
- [Optimized SVD](../optimized_method/README.md) - Efficient SVD computation
- [Mathematical Foundation](../mathematical_foundation/README.md) - SVD theory and implementation from scratch
