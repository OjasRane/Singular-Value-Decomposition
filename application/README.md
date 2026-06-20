# Application Directory

This directory contains the core application utilities for image compression using Singular Value Decomposition (SVD). The main module, `image_compression.py`, provides high-level functions to compress both grayscale and color images using either the optimized NumPy-based SVD implementation or the from-scratch mathematical implementation.

## Files

- `image_compression.py`: Main module with image compression functions.
- `README.md`: This documentation file.

## Module: image_compression.py

This module provides utilities for compressing images using SVD. It supports both grayscale (2D arrays) and color images (3D arrays) by processing each channel separately for color images.

### Functions

#### `optimized_compress(img, k)`

Compresses an image using the optimized SVD implementation.

**Parameters:**
- `img` (numpy.ndarray): Input image as a 2D (grayscale) or 3D (color) array.
- `k` (int): Number of singular values to keep for compression.

**Returns:**
- `compressed_image` (numpy.ndarray): Compressed image with the same dimensions as input. For color images, returned image is in format BGR.

**Raises:**
- `TypeError`: If input is not a 2D or 3D array.

**Notes:**
- For grayscale images, no clipping or type conversion occurs. If needed it needs to be done outside of the function.
- For color images, compresses each BGR channel separately.
- Clips pixel values to [0, 255] and converts to uint8 for color images.

#### `compress_from_scratch(img, k)`

Compresses an image using the from-scratch SVD implementation.

**Parameters:**
- `img` (numpy.ndarray): Input image as a 2D (grayscale) or 3D (color) array.
- `k` (int): Number of singular values to keep for compression.

**Returns:**
- `compressed_image` (numpy.ndarray): Compressed image with the same dimensions as input. For color images, returned image is in format BGR.

**Raises:**
- `TypeError`: If input is not a 2D or 3D array.

**Notes:**
- For grayscale images, no clipping or type conversion occurs. If needed it needs to be done outside of the function.
- For color images, compresses each BGR channel separately.
- Clips pixel values to [0, 255] and converts to uint8 for color images.

#### `get_k_from_compression_ratio(image_shape, compression_ratio, percentage=False)`

Calculates the number of singular values (k) needed to achieve a target compression ratio.

**Parameters:**
- `image_shape` (tuple): Shape of the image (height, width) or (height, width, channels).
- `compression_ratio` (float): Target compression ratio.
- `percentage` (bool, optional): If True, treats compression_ratio as a percentage (0-100). Default is False.

**Returns:**
- `k` (int): Number of singular values to use for the desired compression ratio.

**Notes:**
- The compression ratio is calculated based on the storage requirements of the SVD components.
- Formula used for calculating $k$:
$$\begin{aligned}
k=\text{Compression ratio}\times\frac{mn}{m+n+1}\quad&\text{Compression ratio}\in[0,1]\\
&\text{Image dimensions}=m\times n\\
k=\frac{\text{Compression ratio}}{100}\times\frac{mn}{m+n+1}\quad&\text{Compression ratio}\in[0,100]\%\\
&\text{Image dimensions}=m\times n
\end{aligned}$$

### Class: `SVDCompressor`

A utility class that decomposes an image once and allows repeated reconstructions with different values of `k` without recomputing the SVD. This is particularly useful for interactive applications such as Streamlit, where users may frequently adjust compression parameters.

#### `SVDCompressor(image)`

Creates a compressor object for a grayscale or color image.

**Parameters:**
- `image` (`numpy.ndarray`): Input image as a 2D (grayscale) or 3D (color) array.

**Raises:**
- `TypeError`: If the input is not a NumPy array.

**Notes:**
- Supports both grayscale and color images.
- Stores decomposed SVD components internally for later reconstruction.

---

#### `decompose(dtype=np.float64)`

Computes and stores the SVD decomposition of the image.

**Parameters:**
- `dtype` (`numpy.dtype`, optional): Data type used to store the decomposed matrices. Default is `numpy.float64`.

**Returns:**
- `None`

**Notes:**
- For grayscale images, stores a single `(U, S, Vt)` decomposition.
- For color images, each BGR channel is decomposed separately and stored internally.
- This operation is computationally expensive and is intended to be performed only once per image.

---

#### `reconstruct(k=None)`

Reconstructs the image using the previously stored decomposition.

**Parameters:**
- `k` (`int`, optional): Number of singular values to retain. If `None`, all singular values are used.

**Returns:**
- `reconstructed_image` (`numpy.ndarray`): Reconstructed image in `uint8` format.

**Notes:**
- For grayscale images, pixel values are clipped to `[0, 255]` and converted to `uint8`.
- For color images, each BGR channel is reconstructed separately, clipped to `[0, 255]`, converted to `uint8`, and merged back into a BGR image.
- Calling this method multiple times with different values of `k` avoids recomputing the SVD, making it significantly faster than repeatedly using `optimized_compress()`.

**Performance Tip:** When building interactive applications (e.g., Streamlit sliders for selecting `k`), use `SVDCompressor` instead of repeatedly calling `optimized_compress()`. The decomposition is computed only once, and subsequent reconstructions are much faster.

## Usage Examples

```python
import cv2
import numpy as np
from image_compression import optimized_compress, compress_from_scratch, get_k_from_compression_ratio, SVDCompressor

# Load an image
img = cv2.imread('path/to/image.jpg')

# Compress using optimized method with k=50
compressed = optimized_compress(img, 50)

# Compress using from-scratch method with k=50
compressed_scratch = compress_from_scratch(img, 50)

# Calculate k for 50% compression ratio
k = get_k_from_compression_ratio(img.shape, 0.5)
compressed_img = optimized_compress(img, k)

# Equivalent using percentage=True
k = get_k_from_compression_ratio(img.shape, 50, percentage=True)
compressed_img = optimized_compress(img, k)

# Compress using SVDCompressor
compressor = SVDCompressor(img)

# Compute SVD once
compressor.decompose()

# Reconstruct with different k values
compressed_50 = compressor.reconstruct(k=50)
compressed_100 = compressor.reconstruct(k=100)
compressed_150 = compressor.reconstruct(k=150)
```

## Dependencies

This module depends on:
- `numpy`: For array operations.
- `opencv-python`: For image processing (splitting/merging channels).
- Local modules: `optimized_method.optimized_svd` and `mathematical_foundation.svd_from_scratch`.

See the main project `requirements.txt` for full dependency list.

## See Also

- [Optimized SVD](../optimized_method/README.md) - Leveraged to perform SVD computations.
- [Mathematical Foundation](../mathematical_foundation/README.md) - Core SVD theory and scratch implementation.
- [Metrics Module](../metrics/README.md) - Utilities to evaluate reconstruction error.
- [SVD Encoder](../svd_encoder/README.md) - Custom binary encoder/decoder for saving compressed images as `.svd` files.