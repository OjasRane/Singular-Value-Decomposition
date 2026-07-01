# Analysis Directory

This directory contains analytical tools and Jupyter notebooks for exploring and understanding Singular Value Decomposition (SVD) concepts, particularly in the context of image compression and Principal Component Analysis (PCA).

## Contents

### `understanding_svd_with_image_compression.ipynb`

This interactive Jupyter notebook provides a comprehensive exploration of SVD fundamentals and their application to image compression. The notebook covers:

- **Mathematical Foundations**: Review of linear algebra concepts underlying SVD.
- **SVD Decomposition**: Step-by-step breakdown of how SVD works.
- **Image Compression Theory**: How SVD can be used to compress images by retaining only the most significant singular values.
- **Visual Demonstrations**: Interactive plots showing the effects of different compression levels.
- **Performance Analysis**: Comparison of compression ratios vs. reconstruction quality.
- **Practical Examples**: Real-world application on sample images.

### `pca_using_svd.ipynb`

This interactive Jupyter notebook demonstrates the application of Singular Value Decomposition (SVD) to Principal Component Analysis (PCA). The notebook covers:

- **PCA Theory**: Core mathematical explanations of dimensionality reduction by projecting data onto maximum variance directions.
- **SVD to PCA Connection**: Exploring how running SVD on mean-centered data directly yields the principal components without calculating the covariance matrix.
- **Visual Demonstrations**: Projecting a synthetic 2D dataset onto 1D principal component axes.
- **Reconstruction Analysis**: Visualizing dataset reconstruction quality after low-rank approximations.

## Prerequisites

Before running the notebook, ensure you have the required dependencies installed:

```bash
pip install -r ../requirements.txt
```

Key dependencies for analysis:
- `numpy` - For numerical computations
- `matplotlib` - For plotting and visualizations
- `opencv-python` - For image processing
- `jupyterlab` - For running the notebook

## Getting Started

1. **Activate the virtual environment** (if using one):

    For Windows
   ```ps1
   .venv\Scripts\Activate
   ```
   For Linux/MacOS
   ```bash
   source .venv/bin/activate
   ```

2. **Launch Jupyter Lab**:
   ```bash
   jupyter lab
   ```

3. **Open the notebook**:
   Navigate to `analysis/understanding_svd_with_image_compression.ipynb` or `analysis/pca_using_svd.ipynb` in the Jupyter interface

4. **Run the cells**:
   Notebook is made by keeping in mind that cells would be running in sequential order, so running cells in sequential order will protect you from any type of error or wrong results.

> ⚠️ **Important Note**:
>   Many variables were re-used, over the notebook same variable may have different type, so if you change any parameter please restart the kernel and re-run the notebook to obtain the updated result.

## Learning Objectives

After completing this analysis, you should understand:

- How SVD decomposes matrices into singular values and vectors.
- The relationship between singular values and information content.
- How SVD is mathematically connected to Principal Component Analysis (PCA).
- How to apply SVD for dimensionality reduction, image compression, and PCA.
- Trade-offs between compression ratio/feature reduction and data reconstruction quality.
- Practical implementation considerations.

## Integration with Project

This analysis complements the other components:

- **Mathematical Foundation**: Provides theoretical background for the from-scratch implementation. Module used: `svd_from_scratch.py`.
- **Application**: Demonstrates practical usage of the compression and PCA utilities. Modules used: `image_compression.py` and `principal_component_analysis.py`.
- **Metrics**: Uses evaluation functions to assess compression quality. Module used: `metrics.py`.
- **Optimized Method**: Compares with high-performance implementations. Module used: `optimized_svd.py`.
- **Web App**: SVD image compression, custom image/SVD viewer, and interactive PCA projection playground.

## Contributing

When adding new analysis notebooks or scripts:

1. Follow the existing naming conventions
2. Include comprehensive documentation within the notebook
3. Add any new dependencies to `../requirements.txt`
4. Update this README with the new content description