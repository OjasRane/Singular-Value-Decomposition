from optimized_method import optimized_svd
from mathematical_foundation import svd_from_scratch

def optimized_compress(img, k):
    U, S, Vt = optimized_svd.optimized_svd(img)
    compressed_image = optimized_svd.reconstruct(U, S, Vt, k)
    return compressed_image

def compress_from_scratch(img, k):
    U, S, Vt = svd_from_scratch.svd_from_scratch(img)
    if k is not None:
        U = U[:, :k]
        S = S[:k, :k]
        Vt = Vt[:k, :]

    compressed_image =  U @ S @ Vt
    return compressed_image