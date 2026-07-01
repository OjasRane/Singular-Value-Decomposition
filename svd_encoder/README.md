# SVD Encoder

This directory contains the SVD encoding and decoding module, which implements a custom binary file format (`.svd`) to efficiently store and reconstruct compressed images using Singular Value Decomposition.

## Files

- `svd.py`: Core module containing functions for exporting images to the custom `.svd` binary format and reading/reconstructing them.
- `README.md`: This documentation file.

## Module: svd.py

This module provides utilities to save compressed images to disk using a custom binary representation and reconstruct them. By saving only the truncated SVD component matrices ($U$, $S$, and $V^T$), it reduces disk space usage compared to standard uncompressed formats.

### Functions

#### `export_svd(img, filename, k, grayscale, version=1)`

Encodes and saves an image to a `.svd` file.

**Parameters:**
- `img` (`str`): Path to the input image file.
- `filename` (`str`): Path where the compressed `.svd` file will be saved.
- `k` (`int`): Number of singular values (rank) to retain.
- `grayscale` (`bool`): If `True`, processes the image as grayscale. If `False`, processes it as a color image.
- `version` (`int`, optional): Version of the file format specification. Default is `1`.

**Raises:**
- `FileNotFoundError`: If the input image cannot be loaded.
- `ValueError`: If the value of `k` is invalid (e.g. less than 1 or greater than the image's minimum dimension).

**Notes:**
- For color images, the image is split into its Blue, Green, and Red channels, and SVD decomposition is performed on each channel separately.
- Matrix components are cast to 32-bit floats (`np.float32`) for storage.

---

#### `read_svd(filename)`

Reads a `.svd` file and reconstructs the image from its SVD components.

**Parameters:**
- `filename` (`str`): Path to the `.svd` file.

**Returns:**
- `image` (`numpy.ndarray`): Reconstructed image as a NumPy array (in BGR format for color images, or 2D grayscale array).
- `grayscale` (`bool`): `True` if the reconstructed image was grayscale, `False` if it was color.

**Raises:**
- `ValueError`: If the file has incorrect magic bytes (`SVD` at start, `QED` at end), an unsupported version, or if the file content is truncated or corrupted.

**Notes:**
- For color images, the blue, green, and red channels are reconstructed independently, clipped to the valid range `[0, 255]`, converted back to 8-bit unsigned integers (`uint8`), and merged to produce the final BGR image.

## Binary Format Specification

The `.svd` file format uses a structured binary layout to store the components of the truncated Singular Value Decomposition.

### File Layout

| Field | Size (Bytes) | Data Type / Value | Description |
|---|---|---|---|
| **Magic Header** | 3 | ASCII string `SVD` | Identifies the file format |
| **Version** | 1 | 8-bit unsigned integer | Format version (currently `1`) |
| **Mode** | 1 | ASCII character `G` or `C` | `G` for Grayscale, `C` for Color |
| **Height ($m$)** | 4 | 32-bit little-endian integer | Original image height in pixels |
| **Width ($n$)** | 4 | 32-bit little-endian integer | Original image width in pixels |
| **Rank ($k$)** | 4 | 32-bit little-endian integer | Number of singular values retained |
| **Payload** | Variable | `float32` arrays | Truncated SVD matrices |
| **Magic Footer** | 3 | ASCII string `QED` | Identifies the end of a valid file |

### Payload Layout

#### Grayscale Mode (`G`):
The payload consists of three matrices written consecutively:
1. **$U$ Matrix**: Shape $m \times k$ (Size: $m \times k \times 4$ bytes)
2. **$S$ Vector**: Length $k$ (Size: $k \times 4$ bytes)
3. **$V^T$ Matrix**: Shape $k \times n$ (Size: $k \times n \times 4$ bytes)

#### Color Mode (`C`):
The payload contains the SVD components for the Blue, Green, and Red channels stored sequentially:
$$[U_B, S_B, V^T_B, U_G, S_G, V^T_G, U_R, S_R, V^T_R]$$

Each channel contains:
1. **$U$ Matrix**: Shape $m \times k$ (Size: $m \times k \times 4$ bytes)
2. **$S$ Vector**: Length $k$ (Size: $k \times 4$ bytes)
3. **$V^T$ Matrix**: Shape $k \times n$ (Size: $k \times n \times 4$ bytes)

## Usage Examples

### Encoding and Decoding an Image

```python
import cv2 as cv
from svd_encoder.svd import export_svd, read_svd

# Define paths
input_image_path = "data/image.png"
compressed_svd_path = "compressed_image.svd"

# 1. Export the image using SVD encoding
# Retain 50 singular values, encoding as a color image
export_svd(
    img=input_image_path,
    filename=compressed_svd_path,
    k=50,
    grayscale=False
)

# 2. Read and reconstruct the image from the .svd file
reconstructed_img, is_grayscale = read_svd(compressed_svd_path)

# 3. Save or display the reconstructed image
cv.imwrite("reconstructed_image.png", reconstructed_img)
```

## Dependencies

This module depends on:
- `numpy`: For array reshaping and handling binary data conversion.
- `opencv-python`: For loading, splitting, merging, and saving image data.
- Local modules: `optimized_method.optimized_svd` and `utils`.

See the main project `requirements.txt` for the full dependency list.

## See Also

- [Optimized SVD](../optimized_method/README.md) - Leveraged to perform the SVD computations.
- [Application Directory](../application/README.md) - High-level SVD compression and PCA routines.
- [Metrics Module](../metrics/README.md) - Utilities to evaluate reconstruction error and compression efficiency.
- [Mathematical Foundation](../mathematical_foundation/README.md) - Learn how SVD works under the hood.
- [Web App](../web_app/README.md) - Streamlit interactive web application for image compression playground, compressor, and `.svd` viewer.
