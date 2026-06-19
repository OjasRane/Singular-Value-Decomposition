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

def get_k_from_compression_ratio(image_shape, compression_ratio, percentage=False):
    if percentage:
        k = int((compression_ratio / 100) * (np.prod(image_shape[:2]) / (1 + np.sum(image_shape[:2]))))
    else:
        k = int(compression_ratio * (np.prod(image_shape[:2]) / (1 + np.sum(image_shape[:2]))))
    return k

class SVDCompressor:
    """
    This class is made to implement the optimized method for image compression.
    It is specially made for streamlit app.
    """
    def __init__(self, image):
        if isinstance(image, np.ndarray):
                self.image = image
                self.dim = image.ndim
        else:
            raise TypeError("A ndarray of 2 or 3 dimension was expected")
        self.decomposed_image = None
        self.decomposed_B = None
        self.decomposed_G = None
        self.decomposed_R = None

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