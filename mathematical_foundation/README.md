# Mathematical Foundation

This directory contains the mathematical foundation for the Singular Value Decomposition (SVD) implementation used in this project.

## The mathematics
The naive approach computes $U$ and $V$ independently, which can produce inconsistent eigenvectors. The implementation here avoids this by deriving $U$ directly from $V$.
### Algorithm used in `svd_from_scratch` function

$$\begin{aligned}
&\text{Form }V\text{ matrix using }A^{T}A\\
&\text{Form }\Sigma\text{ matrix using eigenvalues of }A^{T}A\\
&\text{Consider matrix }V\text{ as:}\\
&V=\begin{bmatrix}v_{1}\\v_{2}\\\vdots\\v_{i}\\\vdots\\v_{n}\end{bmatrix}\quad V\in\mathbb{R}^{n\times n};\quad v_{i}\in\mathbb{R}^{1\times n}\\
&\text{and matrix }U\text{ as:}\\
&U=\begin{bmatrix}u_{1}&u_{2}&\cdots&u_{i}&\cdots&u_{m}\end{bmatrix}\quad U\in\mathbb{R}^{m\times m};\quad u_{i}\in\mathbb{R}^{m\times1}\\
&\text{Relation between }u_{i}\text{ and }v_{i}:\\
&\text{We know that }A=U\Sigma V^{T}\quad V\text{ is orthogonal matrix, hence; }VV^{T}=I\\
&AV=U\Sigma\\
&A\begin{bmatrix}v_{1}&v_{2}&\cdots&v_{i}&\cdots&v_{n}\end{bmatrix}=\begin{bmatrix}u_{1}&u_{2}&\cdots&u_{i}&\cdots&u_{m}\end{bmatrix}\text{diag}(\sigma_{1},\sigma_{2},\cdots,\sigma_{i},\cdots,\sigma_{\min{(m, n)}})\\
\therefore&Av_{i}=\sigma_{i}u_{i}\\
&u_{i}=\frac{Av_{i}}{\sigma_{i}}\quad\sigma_{i}=i^{\text{th}}\text{ singular value of matrix }A
\end{aligned}$$

## Contents

- `svd_from_scratch.py` - A minimal Python implementation of SVD built from linear algebra principles.

## Purpose

The goal of this folder is to demonstrate the core mathematical steps needed to compute SVD without relying on a high-level library function such as `numpy.linalg.svd`.

This implementation is useful for learning the underlying mechanics of SVD and how the decomposition can be derived from eigenvalue problems.

## How it works

The `svd_from_scratch` function follows these general steps:

1. Normalize the input matrix by its largest element for numerical stability.
2. Compute the symmetric matrix `A^T A`.
3. Solve the eigenvalue problem for `A^T A` to obtain eigenvalues and eigenvectors.
4. Sort eigenvalues and eigenvectors in descending order.
5. Construct the matrix `V` from the eigenvectors.
6. Form the singular value matrix `S` using the square roots of the eigenvalues.
7. Compute the left-singular vectors `U` by projecting the input matrix onto the right-singular vectors and dividing by singular values.
8. Scale the result back to the original input magnitude.

## Usage

```python
from mathematical_foundation.svd_from_scratch import svd_from_scratch

A = np.array([[1.0, 2.0], [3.0, 4.0]])
U, S, Vt = svd_from_scratch(A)

print("U:\n", U)
print("S:\n", S)
print("V^T:\n", Vt)
```

## Notes

- This implementation is intended for educational purposes and may not be as numerically stable or efficient as optimized library routines.
- The matrix `S` is returned with the same shape as the original input matrix.
- `V.T` is returned as `Vt` to match the usual SVD notation `A = U S V^T`.

## Related files

- `application/image_compression.py` - Uses SVD to compress images.
- `optimized_method/optimized_svd.py` - Contains a more efficient SVD implementation built around optimized NumPy operations.
- `metrics/metrics.py` - Provides metrics for evaluating SVD-based reconstruction quality.
