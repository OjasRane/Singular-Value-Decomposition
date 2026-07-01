# SVD Lab - Web Application

A Streamlit-based interactive web application for exploring and experimenting with Singular Value Decomposition (SVD) applied to image compression and Principal Component Analysis (PCA).

## Overview

SVD Lab is an interactive platform that demonstrates how Singular Value Decomposition can be used to compress images with minimal quality loss, and visualizes how SVD is applied to project and reconstruct datasets using PCA. The application provides a playground mode for both image compression and PCA experimentation, a compressor tool for practical image compression, and a built-in image viewer capable of loading standard images or reconstructing custom `.svd` files on the fly.

## Features

- **Playground**: Interactive modes to experiment with SVD image compression parameters in real-time, and visualize Principal Component Analysis (PCA) projection axes using synthetic datasets
- **Compressor**: Upload your own images and compress them using SVD with adjustable compression parameters
- **Image Viewer**: View standard images (PNG, JPG, JPEG) or custom `.svd` binary format files directly in the web browser
- **Grayscale/Color Support**: Support for both grayscale and color image compression
- **Interactive Visualization**: Real-time visualization of compressed images and metrics
- **Educational Resources**: Links to detailed documentation and mathematical explanations
- **Download Capability**: Download compressed images in PNG, JPEG, or custom `.svd` format

## Project Structure

```
web_app/
├── .streamlit/
│   └── config.toml         # Streamlit configuration
├── landing_page.py          # Main landing page with navigation
├── pages/
│   ├── compressor.py        # Image compression tool
│   ├── image_viewer.py      # Standard and .svd image viewer
│   └── playground.py        # Interactive playground mode
└── README.md
```

## Installation

### Prerequisites

- Python 3.8 or higher
- See [requirements.txt](../requirements.txt) in the project root

### Setup

Navigate to the project root directory:
```bash
cd Singular-Value-Decomposition
```

#### Using uv:
```
uv sync
```

#### Using pip:
1. Create a virtual environment (Ignore is already done):
   ```bash
   python -m venv .venv                         # For Windows
   python3 -m venv .venv                        # For MacOS/Linux
   ```

2. Activate the virtual environment:
   ```bash
   .venv\Scripts\Activate.ps1                   # For Windows
   source .venv/bin/activate                    # For MacOS/Linux
   ```

3. Installing dependencies:
   ```bash
   pip install -e .
   ```

## Running the Application

From the project root directory, start the Streamlit app:

### Using uv:
```bash
uv run streamlit run web_app\landing_page.py    # For Windows
uv run streamlit run web_app/landing_page.py    # For MacOS/Linux
```
### Using pip:
```bash
streamlit run web_app\landing_page.py           # For Windows
streamlit run web_app/landing_page.py           # For MacOS/Linux
```

The application will open in your default web browser (typically http://localhost:8501)

## Pages Overview

### Landing Page (`landing_page.py`)
The main entry point featuring navigation sections:

- **Playground**: Jump into interactive SVD compression experiments
- **Compressor**: Upload and compress your own images
- **Image Viewer**: View standard images or custom SVD compressed files
- **Documentation**: Link to the detailed mathematical explanation notebook
- **GitHub**: Link to the project repository
- **Connect**: Social media links

### Playground (`pages/playground.py`)
Interactive mode supporting two choices of application:

#### 1. Image Compression
- Pre-loaded sample image for quick testing.
- Toggle between grayscale and color modes.
- Adjustable compression parameter (`k`) using a slider.
- Real-time visualization of compressed results.
- Interactive plots for reconstruction error and energy retained.
- Home button to return to landing page.

#### 2. Principal Component Analysis
- Generate a custom synthetic 2D normal distribution dataset by controlling the random seed, number of samples, X/Y axes scaling, and rotation angle.
- Select from three visual plots: Original Dataset, Original Dataset with Principal Axes, and Data points projected on Principal Axes.
- Control the number of projection dashed lines to visualize the orthogonal distance to the principal axes.

Features:
- Cached calculations for performance.
- Interactive controls for real-time parameter tuning.
- Visual feedback on compression quality and PCA projections.

### Compressor (`pages/compressor.py`)
Practical tool for compressing your own images:

- File uploader for PNG, JPG, and JPEG formats
- Grayscale/color toggle
- `k` input prefilled using a default 0.25 compression-ratio estimate
- Compressed image preview
- Download options (PNG, JPEG or SVD format)
- Home button for navigation

Usage:
1. Upload an image file
2. Choose grayscale or color mode
3. Adjust the `k` parameter (higher = better quality, larger file)
4. Verify the compressed result
5. Download in your preferred format (PNG, JPEG, or custom `.svd` format)

### Image Viewer (`pages/image_viewer.py`)
Utility to view images directly in your browser:

- Support for standard image files: PNG, JPG, JPEG
- Support for custom `.svd` binary compressed files (uses the `svd_encoder` module to reconstruct and view on the fly)
- Display size adjusted to fit content
- Home button to return to landing page

## How SVD Image Compression Works

SVD decomposes an image matrix into three matrices: $U$, $\Sigma$, and $V^{T}$. By keeping only the top-k singular values and their corresponding vectors, we reduce the data needed to represent the image while preserving visual quality.

For detailed mathematical explanations, see [analysis/understanding_svd_with_image_compression.ipynb](../analysis/understanding_svd_with_image_compression.ipynb).

## How SVD is Applied to Principal Component Analysis (PCA)

PCA identifies principal component directions (axes of maximum variance) by centering the input dataset and computing its SVD: $X - \mu = U \Sigma V^T$. The columns of $V$ represent the principal components (axes), and singular values in $\Sigma$ directly determine the explained variance ratio along each axis.

For detailed mathematical explanations, see [analysis/pca_using_svd.ipynb](../analysis/pca_using_svd.ipynb).

## Dependencies

The web app depends on:

- **streamlit**: Web framework
- **numpy**: Numerical computations
- **opencv-python**: Image processing
- **pillow**: Image format handling
- **matplotlib**: Visualization

From the parent project:
- `application/image_compression.py`: Compression utilities
- `application/principal_component_analysis.py`: PCA implementation utilities
- `metrics/metrics.py`: Quality metrics
- `svd_encoder/svd.py`: SVD decoder for loading `.svd` compressed files

## Tips for Best Results

- **Playground**: Start with low k values and gradually increase to see compression effects
- **Compressor**: Use the 0.25 (25%) default compression ratio for a good balance
- **Color Images**: Color compression will produce larger files than grayscale (3 channels)
- **File Formats**: For best compression analysis, download as PNG or custom `.svd` format (lossless in SVD terms) to avoid lossy JPEG re-compression

## Troubleshooting

- **Image not loading**: Ensure the image file is in a supported format (PNG, JPG, JPEG)
- **Performance issues**: Reduce the image size or use grayscale mode for faster processing
- **Missing data file in playground**: Verify that `data/image.png` exists in the project root

## Related Resources

- [Project README](../README.md) - Overall project documentation
- [Application Module](../application/) - Image compression and PCA utilities
- [Mathematical Foundation](../mathematical_foundation/) - SVD implementation details
- [Analysis Notebooks](../analysis/) - Educational notebooks with detailed explanations for SVD compression and PCA
- [SVD Encoder](../svd_encoder/) - Custom binary SVD encoder and decoder
- [SVD Lab](https://svdlab.streamlit.app) - Live deployed app on Streamlit Community Cloud

## License

See [LICENSE](../LICENSE) in the project root.
