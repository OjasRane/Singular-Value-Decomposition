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
        compressed_G = optimized_compress(G, k)
        compressed_R = optimized_compress(R, k)
        compressed_image = cv.merge([np.clip(compressed_B, 0, 255),
                                     np.clip(compressed_G, 0, 255),
                                     np.clip(compressed_R, 0, 255)]).astype("uint8")
    else:
        raise TypeError("A ndarray of 2 or 3 dimension was expected")
    return compressed_image

def compress_from_scratch(img, k):
    if img.ndim == 2:
        U, S, Vt = svd_from_scratch.svd_from_scratch(img)
        compressed_image = optimized_svd.reconstruct(U, S, Vt, k)
    elif img.ndim == 3:
        B, G, R = cv.split(img)
        compressed_B = compress_from_scratch(B)
        compressed_G = compress_from_scratch(G)
        compressed_R = compress_from_scratch(R)
        compressed_image = cv.merge([np.clip(compressed_B, 0, 255),
                                    np.clip(compressed_G, 0, 255),
                                    np.clip(compressed_R, 0, 255)]).astype("uint8")
    else:
        raise TypeError("A ndarray of 2 or 3 dimension was expected")
    return compressed_image

def get_k_from_compression_ratio(image_shape, compression_ratio, percentage=False):
    return int(compression_ratio * (np.prod(image_shape[:2]) / (1 + np.sum(image_shape[:2]))))