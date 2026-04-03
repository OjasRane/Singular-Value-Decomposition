# SVD Lab - Web Application

A Streamlit-based interactive web application for exploring and experimenting with Singular Value Decomposition (SVD) applied to image compression.

## Overview

SVD Lab is an interactive platform that demonstrates how Singular Value Decomposition can be used to compress images with minimal quality loss. The application provides both a playground mode for experimentation and a compressor tool for practical image compression.

## Features

- **Playground**: Interactive mode with a pre-loaded image to experiment with SVD compression parameters in real-time
- **Compressor**: Upload your own images and compress them using SVD with adjustable compression parameters
- **Grayscale/Color Support**: Support for both grayscale and color image compression
- **Interactive Visualization**: Real-time visualization of compressed images and metrics
- **Educational Resources**: Links to detailed documentation and mathematical explanations
- **Download Capability**: Download compressed images in PNG or JPEG format

## Project Structure

```
web_app/
├── landing_page.py          # Main landing page with navigation
├── pages/
│   ├── compressor.py        # Image compression tool
│   ├── playground.py        # Interactive playground mode
├── assets/                  # Static assets (images, icons, etc.)
└── README.md               # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- See [requirements.txt](../requirements.txt) in the project root

### Setup

1. Navigate to the project root directory:
   ```bash
   cd Singular-Value-Decomposition
   ```

2. Create and activate a virtual environment:
   
   Ignore if already done.

   For Windows:
   ```bash
   python -m venv venv
   venv\Scripts\Activate.ps1 
   ```
   For MacOS/Linux:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

From the project root directory, start the Streamlit app:

```bash
streamlit run web_app/landing_page.py
```

The application will open in your default web browser (typically http://localhost:8501)

## Pages Overview

### Landing Page (`landing_page.py`)
The main entry point featuring four sections:

- **Playground**: Jump into interactive SVD compression experiments
- **Compressor**: Upload and compress your own images
- **Documentation**: Link to the detailed mathematical explanation notebook
- **GitHub**: Link to the project repository
- **Connect**: Social media links

### Playground (`pages/playground.py`)
Interactive mode for experimenting with SVD compression:

- Pre-loaded sample image for quick testing
- Toggle between grayscale and color modes
- Adjustable compression parameter (`k`) using a slider
- Real-time visualization of compressed results
- Metrics display (compression ratio, reconstruction error)
- Home button to return to landing page

Features:
- Cached SVD computation for performance
- Interactive controls for parameter tuning
- Visual feedback on compression quality

### Compressor (`pages/compressor.py`)
Practical tool for compressing your own images:

- File uploader for PNG, JPG, and JPEG formats
- Grayscale/color toggle
- Automatic `k` parameter calculation based on desired compression ratio
- Side-by-side image comparison
- Download options (PNG or JPEG format)
- Home button for navigation

Usage:
1. Upload an image file
2. Choose grayscale or color mode
3. Adjust the `k` parameter (higher = better quality, larger file)
4. Verify the compressed result
5. Download in your preferred format

## How SVD Image Compression Works

SVD decomposes an image matrix into three matrices: $U$, $\Sigma$, and $V^{T}$. By keeping only the top-k singular values and their corresponding vectors, we reduce the data needed to represent the image while preserving visual quality.

For detailed mathematical explanations, see:
- [analysis/understanding_svd_with_image_compression.ipynb](../analysis/understanding_svd_with_image_compression.ipynb)
- Or visit the notebook link in the application

## Dependencies

The web app depends on:

- **streamlit**: Web framework
- **numpy**: Numerical computations
- **opencv-python**: Image processing
- **pillow**: Image format handling
- **matplotlib**: Visualization

From the parent project:
- `optimized_method/optimized_svd.py`: SVD implementation
- `application/image_compression.py`: Compression utilities
- `metrics/metrics.py`: Quality metrics

## Tips for Best Results

- **Playground**: Start with low k values and gradually increase to see compression effects
- **Compressor**: Use the 0.25 (25%) default compression ratio for a good balance
- **Color Images**: Color compression will produce larger files than grayscale (3 channels)
- **File Formats**: For best compression analysis, download as PNG to avoid JPEG re-compression

## Troubleshooting

- **Image not loading**: Ensure the image file is in a supported format (PNG, JPG, JPEG)
- **Performance issues**: Reduce the image size or use grayscale mode for faster processing
- **Missing data file in playground**: Verify that `data/image.png` exists in the project root

## Related Resources

- [Project README](../README.md) - Overall project documentation
- [Application Module](../application/) - Image compression utilities
- [Mathematical Foundation](../mathematical_foundation/) - SVD implementation details
- [Analysis Notebook](../analysis/) - Educational notebook with detailed explanations

## License

See [LICENSE](../LICENSE) in the project root.
