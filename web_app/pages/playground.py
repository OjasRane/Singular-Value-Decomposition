import streamlit as st
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from application import image_compression
from metrics import metrics
from application.image_compression import get_k_from_compression_ratio

st.set_page_config(
    page_title="SVD Lab | Playground",
    page_icon=":material/grid_4x4:"
)
st.title("Playground", text_alignment="center")

if st.checkbox("Grayscale", value=True, key="grayscale"):
    image = cv.imread(r"data/image.png", cv.IMREAD_GRAYSCALE)
else:
    image = cv.imread(r"data/image.png")

if "previous_grayscale" not in st.session_state:
    st.session_state["previous_grayscale"] = not st.session_state["grayscale"]

k = min(image.shape[:2])

if "k" not in st.session_state:
    st.session_state["k"] = k

if st.session_state["previous_grayscale"] is not st.session_state["grayscale"]:
    st.session_state["compressor_playground"] = image_compression.SVDCompressor(image)
    st.session_state["compressor_playground"].decompose()
    st.session_state["previous_grayscale"] = st.session_state["grayscale"]

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
    if st.session_state["grayscale"]:
        compressed_image = st.session_state["compressor_playground"].reconstruct(st.session_state["k"])
        st.image(compressed_image, caption=f"""Reconstructed Image
                                               \nCompression Ratio={np.round(metrics.compression_ratio(image.shape, st.session_state['k']), 2)}""")
    else:
        compressed_image = st.session_state["compressor_playground"].reconstruct(st.session_state["k"])
        st.image(compressed_image, caption=f"""Reconstructed Image
                                                       \nCompression Ratio={np.round(metrics.compression_ratio(image.shape, st.session_state['k']), 2)}""",
                 channels="BGR")

with st.container(border=True, key="info_container"):
    st.markdown(f"For $k<{get_k_from_compression_ratio(image.shape, 1)}$ compression is feasible else the size increases! (True if data types are ignored)",
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
            error[i-1] = metrics.reconstruction_error(st.session_state["compressor_playground"].decomposed_image[1], i)
    else:
        for i in kx:
            error[i-1] = np.sqrt(metrics.reconstruction_error_squared(st.session_state["compressor_playground"].decomposed_B[1], i) +
                                 metrics.reconstruction_error_squared(st.session_state["compressor_playground"].decomposed_G[1], i) +
                                 metrics.reconstruction_error_squared(st.session_state["compressor_playground"].decomposed_R[1], i))

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
            energy_retained[i-1] = metrics.energy_retained(st.session_state["compressor_playground"].decomposed_image[1], i)
    else:
        for i in kx:
            energy_retained[i-1] = (np.sum(st.session_state["compressor_playground"].decomposed_B[1][:i]**2 + st.session_state["compressor_playground"].decomposed_G[1][:i]**2 + st.session_state["compressor_playground"].decomposed_R[1][:i]**2)) / (np.sum(st.session_state["compressor_playground"].decomposed_B[1]**2 + st.session_state["compressor_playground"].decomposed_G[1]**2 + st.session_state["compressor_playground"].decomposed_R[1]**2))

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