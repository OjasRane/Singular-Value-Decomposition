import cv2 as cv
import numpy as np
from optimized_method import optimized_svd
from mathematical_foundation import svd_from_scratch

def optimized_compress(img, k):
    if img.ndim == 2:
        U, S, Vt = optimized_svd.optimized_svd(img)
        compressed_image = optimized_svd.reconstruct(U, S, Vt, k)
    elif img.ndim == 3:
        B, G, R = cv.split(img)
        compressed_B = optimized_compress(B, k)
        compressed_B = np.asarray(compressed_B)
        compressed_G = optimized_compress(G, k)
        compressed_G = np.asarray(compressed_G)
        compressed_R = optimized_compress(R, k)
        compressed_R = np.asarray(compressed_R)
        compressed_image = cv.merge([
            np.clip(compressed_B, 0, 255),
            np.clip(compressed_G, 0, 255),
            np.clip(compressed_R, 0, 255)
        ]).astype("uint8")
    else:
        raise TypeError("A ndarray of 2 or 3 dimension was expected")
    return compressed_image

def compress_from_scratch(img, k):
    if img.ndim == 2:
        U, S, Vt = svd_from_scratch.svd_from_scratch(img)
        compressed_image = optimized_svd.reconstruct(U, S, Vt, k)
    elif img.ndim == 3:
        B, G, R = cv.split(img)
        compressed_B = compress_from_scratch(B, k)
        compressed_B = np.asarray(compressed_B)
        compressed_G = compress_from_scratch(G, k)
        compressed_G = np.asarray(compressed_G)
        compressed_R = compress_from_scratch(R, k)
        compressed_R = np.asarray(compressed_R)
        compressed_image = cv.merge([
            np.clip(compressed_B, 0, 255),
            np.clip(compressed_G, 0, 255),
            np.clip(compressed_R, 0, 255)
        ]).astype("uint8")
    else:
        raise TypeError("A ndarray of 2 or 3 dimension was expected")
    return compressed_image

def get_k_from_compression_ratio(image_shape, compression_ratio, dtype=np.uint8, percentage=False):
    if percentage:
        k = int((compression_ratio / 100) * (np.prod(image_shape[:2]) / ((1 + np.sum(image_shape[:2])) * np.dtype(dtype).itemsize)))
    else:
        k = int(compression_ratio * (np.prod(image_shape[:2]) / ((1 + np.sum(image_shape[:2])) * np.dtype(dtype).itemsize)))
    return k

class SVDCompressor:
    """
    This class is made to implement the optimized method for image compression.
    It is specially made for streamlit app.
    """
    def __init__(self, image):
        if isinstance(image, np.ndarray):
                self.image = image
                self.grayscale = (image.ndim == 2)
        else:
            raise TypeError("A ndarray of 2 or 3 dimension was expected")
        self.m, self.n = image.shape[0], self.image.shape[1]
        self.decomposed_image = None
        self.decomposed_B = None
        self.decomposed_G = None
        self.decomposed_R = None
        self.k = None

    def decompose(self, dtype=np.float64):
        """
        This method decomposes an image in U, S, Vt matrices and stores them. This is useful for streamlit app to improve its performance.
        :param dtype: Specify the dtype in which U, S, Vt should be
        :return: None
        """
        if self.image.ndim == 2:
            self.decomposed_image = optimized_svd.optimized_svd(self.image, dtype=dtype)
        elif self.image.ndim == 3:
            B, G, R = cv.split(self.image)
            self.decomposed_B = optimized_svd.optimized_svd(B, dtype=dtype)
            self.decomposed_G = optimized_svd.optimized_svd(G, dtype=dtype)
            self.decomposed_R = optimized_svd.optimized_svd(R, dtype=dtype)
        else:
            raise TypeError("A ndarray of 2 or 3 dimension was expected")

    def reconstruct(self, k=None):
        """
        This method reconstructs an image which was stored as U, S, Vt.
        :param k: The number of singular values to use for reconstruction.
        :return: Reconstructed Image.
        """
        self.k = k
        if self.image.ndim == 2:
            U, S, Vt = self.decomposed_image
            return np.clip(optimized_svd.reconstruct(U, S, Vt, k), 0, 255).astype("uint8")
        elif self.image.ndim == 3:
            U, S, Vt = self.decomposed_B
            B = np.clip(optimized_svd.reconstruct(U, S, Vt, k), 0, 255).astype("uint8")

            U, S, Vt = self.decomposed_R
            R = np.clip(optimized_svd.reconstruct(U, S, Vt, k), 0, 255).astype("uint8")

            U, S, Vt = self.decomposed_G
            G = np.clip(optimized_svd.reconstruct(U, S, Vt, k), 0, 255).astype("uint8")

            return cv.merge([B, G, R])

    def export_svd(self, buffer, version=1):
        buffer.write("SVD".encode("utf-8"))
        buffer.write(int.to_bytes(version, 1, "little"))
        buffer.write("G".encode("utf-8") if self.grayscale else "C".encode("utf-8"))
        buffer.write(int.to_bytes(self.m, 4, "little"))
        buffer.write(int.to_bytes(self.n, 4, "little"))
        buffer.write(int.to_bytes(self.k, 4, "little"))
        if self.grayscale:
            U, S, Vt = self.decomposed_image
            U = U[:, :self.k]
            buffer.write(U.tobytes())
            S = S[:self.k]
            buffer.write(S.tobytes())
            Vt = Vt[:self.k, :]
            buffer.write(Vt.tobytes())
        else:
            U, S, Vt = self.decomposed_B
            U = U[:, :self.k]
            buffer.write(U.tobytes())
            S = S[:self.k]
            buffer.write(S.tobytes())
            Vt = Vt[:self.k, :]
            buffer.write(Vt.tobytes())

            U, S, Vt = self.decomposed_G
            U = U[:, :self.k]
            buffer.write(U.tobytes())
            S = S[:self.k]
            buffer.write(S.tobytes())
            Vt = Vt[:self.k, :]
            buffer.write(Vt.tobytes())

            U, S, Vt = self.decomposed_R
            U = U[:, :self.k]
            buffer.write(U.tobytes())
            S = S[:self.k]
            buffer.write(S.tobytes())
            Vt = Vt[:self.k, :]
            buffer.write(Vt.tobytes())
        buffer.write("QED".encode("utf-8"))