import numpy as np
import cv2 as cv

from optimized_method import optimized_svd
from utils import read_data

def export_svd(img, filename, k, grayscale, version=1):
    img = cv.imread(img, cv.IMREAD_GRAYSCALE if grayscale else cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError("Image not found")
    m, n = img.shape[0], img.shape[1]
    if k > min(m, n) or k < 1:
        raise ValueError("Invalid value of k")
    if grayscale:
        U, S, Vt = optimized_svd.optimized_svd(img, dtype=np.float32)
        U = U[:, :k]
        S = S[:k]
        Vt = Vt[:k]
    else:
        B, G, R = cv.split(img)
        B_U, B_S, B_Vt = optimized_svd.optimized_svd(B, dtype=np.float32)
        B_U, B_S, B_Vt = B_U[:, :k], B_S[:k], B_Vt[:k]

        G_U, G_S, G_Vt = optimized_svd.optimized_svd(G, dtype=np.float32)
        G_U, G_S, G_Vt = G_U[:, :k], G_S[:k], G_Vt[:k]

        R_U, R_S, R_Vt = optimized_svd.optimized_svd(R, dtype=np.float32)
        R_U, R_S, R_Vt = R_U[:, :k], R_S[:k], R_Vt[:k]

    with open(filename, "wb") as f:
        f.write("SVD".encode("utf-8"))
        f.write(int.to_bytes(version, 1, byteorder="little"))
        if grayscale:
            f.write("G".encode("utf-8"))
        else:
            f.write("C".encode("utf-8"))
        f.write(int.to_bytes(m, 4, byteorder="little"))
        f.write(int.to_bytes(n, 4, byteorder="little"))
        f.write(int.to_bytes(k, 4, byteorder="little"))
        if grayscale:
            U.tofile(f, sep="")
            S.tofile(f, sep="")
            Vt.tofile(f, sep="")
        else:
            B_U.tofile(f, sep="")
            B_S.tofile(f, sep="")
            B_Vt.tofile(f, sep="")
            G_U.tofile(f, sep="")
            G_S.tofile(f, sep="")
            G_Vt.tofile(f, sep="")
            R_U.tofile(f, sep="")
            R_S.tofile(f, sep="")
            R_Vt.tofile(f, sep="")
        f.write("QED".encode("utf-8"))

def read_svd(source):
    if hasattr(source, "read"):
        f = source
    else:
        f = open(source, "rb")
    try:
        f.seek(-3, 2)
        footer = f.read(3)
        f.seek(0)
        magic = f.read(3)
        version = int.from_bytes(f.read(1), byteorder="little")
        grayscale = True if f.read(1) == b'G' else False
        if magic == b'SVD' and footer == b'QED':
            if version == 1:
                m = int.from_bytes(f.read(4), byteorder="little")
                n = int.from_bytes(f.read(4), byteorder="little")
                k = int.from_bytes(f.read(4), byteorder="little")
                U_size = m*k*4
                S_size = k*4
                Vt_size = k*n*4
                if grayscale:
                    U = np.frombuffer(read_data(f, U_size), dtype=np.float32).reshape(m, k)
                    S = np.frombuffer(read_data(f, S_size), dtype=np.float32)
                    Vt = np.frombuffer(read_data(f, Vt_size), dtype=np.float32).reshape(k, n)
                    image = np.clip(optimized_svd.reconstruct(U, S, Vt), 0, 255).astype("uint8")
                    return image, grayscale
                else:
                    B_U = np.frombuffer(read_data(f, U_size), dtype=np.float32).reshape(m, k)
                    B_S = np.frombuffer(read_data(f, S_size), dtype=np.float32)
                    B_Vt = np.frombuffer(read_data(f, Vt_size), dtype=np.float32).reshape(k, n)
                    B = np.clip(optimized_svd.reconstruct(B_U, B_S, B_Vt), 0, 255).astype("uint8")

                    G_U = np.frombuffer(read_data(f, U_size), dtype=np.float32).reshape(m, k)
                    G_S = np.frombuffer(read_data(f, S_size), dtype=np.float32)
                    G_Vt = np.frombuffer(read_data(f, Vt_size), dtype=np.float32).reshape(k, n)
                    G = np.clip(optimized_svd.reconstruct(G_U, G_S, G_Vt), 0, 255).astype("uint8")

                    R_U = np.frombuffer(read_data(f, U_size), dtype=np.float32).reshape(m, k)
                    R_S = np.frombuffer(read_data(f, S_size), dtype=np.float32)
                    R_Vt = np.frombuffer(read_data(f, Vt_size), dtype=np.float32).reshape(k, n)
                    R = np.clip(optimized_svd.reconstruct(R_U, R_S, R_Vt), 0, 255).astype("uint8")

                    image = cv.merge([B, G, R])
                    return image, grayscale

            else:
                raise ValueError("Invalid Version")
        else:
            raise ValueError("Invalid .svd file")
    finally:
        if not hasattr(source, "read"):
            f.close()