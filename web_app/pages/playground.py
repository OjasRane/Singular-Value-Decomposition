import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from optimized_method import optimized_svd
from metrics import metrics
from application.image_compression import get_k_from_compression_ratio

st.set_page_config(
    page_title="SVD Lab | Playground",
    page_icon=":material/grid_4x4:"
)
st.title("Playground", text_alignment="center")

if st.checkbox("Grayscale", value=True, key="grayscale"):
    image = cv.imread(r"data\image.png", cv.IMREAD_GRAYSCALE)
else:
    image = cv.imread(r"data\image.png")

k = min(image.shape[:2])

if "k" not in st.session_state:
    st.session_state["k"] = k

@st.cache_data
def svd(A):
    return optimized_svd.optimized_svd(A)

col1, col2 = st.columns(2)

with col1:
    if image.ndim == 2:
        st.image(image, caption="""Original Image
                                    \nSource: [This person does not exist dot com](https://thispersondoesnotexist.com)""")
    else:
        st.image(image, caption="""Original Image
                                    \nSource: [This person does not exist dot com](https://thispersondoesnotexist.com)""",
                 channels="BGR")

with col2:
    if image.ndim == 2:
        U, S, Vt = svd(image)
        compressed_image = optimized_svd.reconstruct(U, S, Vt, st.session_state["k"])
        compressed_image_display = np.clip(compressed_image, 0, 255).astype("uint8")
        st.image(compressed_image_display, caption=f"""Reconstructed Image
                                               \nCompression Ratio={np.round(metrics.compression_ratio(image.shape, st.session_state['k']), 2)}""")
    else:
        B, G, R = cv.split(image)
        B_U, B_S, B_Vt = svd(B)
        G_U, G_S, G_Vt = svd(G)
        R_U, R_S, R_Vt = svd(R)

        compressed_B = optimized_svd.reconstruct(B_U, B_S, B_Vt, st.session_state["k"])
        compressed_G = optimized_svd.reconstruct(G_U, G_S, G_Vt, st.session_state["k"])
        compressed_R = optimized_svd.reconstruct(R_U, R_S, R_Vt, st.session_state["k"])
        compressed_image = cv.merge([compressed_B, compressed_G, compressed_R])
        compressed_image_display = np.clip(compressed_image, 0, 255).astype("uint8")
        st.image(compressed_image_display, caption=f"""Reconstructed Image
                                                       \nCompression Ratio={np.round(metrics.compression_ratio(image.shape, st.session_state['k']), 2)}""",
                 channels="BGR")

with st.container(border=True, key="info_container"):
    st.markdown(f"For $k<{get_k_from_compression_ratio(image.shape, 1)}$ compression is feasible else the size increases!",
                text_alignment="center")

st.slider("k", min_value=1, max_value=k, value=k, key="k")

kx = np.arange(1, st.session_state["k"]+1)
fig, ax = plt.subplots()
st.radio("Select the plot to view it",
         ["Reconstruction Error", "Energy Retained"],
         key="plot")

if st.session_state["plot"] == "Reconstruction Error":
    error = np.empty(st.session_state["k"])
    if image.ndim == 2:
        for i in kx:
            error[i-1] = metrics.reconstruction_error(S, i)
    else:
        for i in kx:
            error[i-1] = np.sqrt(metrics.reconstruction_error_squared(B_S, i) +
                                 metrics.reconstruction_error_squared(G_S, i) +
                                 metrics.reconstruction_error_squared(R_S, i))

    ax.plot(kx, error)
    ax.set(xlabel="k", ylabel="Reconstruction Error", title="Reconstruction Error")
    ax.hlines(y=error[st.session_state["k"] - 1], xmin=0, xmax=st.session_state["k"], linestyle="--", colors="black")
    ax.plot(st.session_state["k"], error[st.session_state["k"] - 1], ".", color="black")

    st.markdown(r"""
            $$
            \text{Reconstruction Error}=\sqrt{\sum_{i=k}^{\min{(m,n)}}{\sigma_{i}^{2}}}
            $$
            """, text_alignment="center")
else:
    energy_retained = np.empty(st.session_state["k"])
    if image.ndim == 2:
        for i in kx:
            energy_retained[i-1] = metrics.energy_retained(S, i)
    else:
        for i in kx:
            energy_retained[i-1] = (np.sum(B_S[:i]**2 + G_S[:i]**2 + R_S[:i]**2)) / (np.sum(B_S**2 + G_S**2 + R_S**2))

    st.markdown(r"""
    $$
    \text{Energy Retained}=\frac{\sum_{i=1}^{k}{\sigma_{i}^{2}}}{\sum{\sigma_{i}^{2}}}
    $$
    """, text_alignment="center")

    ax.hlines(y=energy_retained[st.session_state["k"] - 1], xmin=0, xmax=st.session_state["k"], linestyle="--", colors="black")
    ax.plot(kx, energy_retained)
    ax.set(xlabel="k", ylabel="Energy Retained", title="Energy Retained")

    ax.plot(st.session_state["k"], energy_retained[st.session_state["k"] - 1], ".", color="black")

ax.grid(True)
st.pyplot(fig)

if st.button("Home", icon=":material/home:"):
   st.switch_page(r"landing_page.py")