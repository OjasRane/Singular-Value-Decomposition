import streamlit as st
import numpy as np
import cv2 as cv
import io
from PIL import Image
from application import image_compression
from metrics.metrics import compression_ratio

st.set_page_config(
    page_title="SVD Lab | Compressor",
    page_icon=":material/grid_4x4:",
)
st.title("Compressor", text_alignment="center")

image = st.file_uploader("Upload your image", type=["png", "jpg", "jpeg"])
st.checkbox("Greyscale", value=True, key="greyscale")

if "previous_greyscale" not in st.session_state:
    st.session_state["previous_greyscale"] = not st.session_state["greyscale"]

if image is not None:
    image_bytes = image.getvalue()

    if st.session_state["greyscale"]:
        image = cv.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv.IMREAD_GRAYSCALE)
    else:
        image = cv.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv.IMREAD_COLOR)

    needs_decompose = (
        "compressor" not in st.session_state or
        st.session_state.get("image_bytes") != image_bytes or
        st.session_state["greyscale"] is not st.session_state["previous_greyscale"]
    )

    if needs_decompose:
        st.session_state["compressor"] = image_compression.SVDCompressor(image)
        m, n = st.session_state["compressor"].m, st.session_state["compressor"].n
        st.session_state["image_shape"] = (m, n)
        st.session_state["compressor"].decompose(dtype=np.float32)
        st.session_state["previous_greyscale"] = st.session_state["greyscale"]
        st.session_state["image_bytes"] = image_bytes

    k = min(image.shape[:2])
    st.number_input("k", min_value=1, max_value=k, value=image_compression.get_k_from_compression_ratio(image.shape,
                                                                                                   0.8, dtype=np.float32), key="k")
    if not st.session_state["greyscale"]:
        compressed_image = cv.cvtColor(st.session_state["compressor"].reconstruct(st.session_state["k"]), cv.COLOR_BGR2RGB)
    else:
        compressed_image = st.session_state["compressor"].reconstruct(st.session_state["k"])

    caption = f"Compressed Image, Compression Ratio: {round(compression_ratio(st.session_state['image_shape'], st.session_state['k'], dtype=np.float32), 4)}"
    st.image(compressed_image, caption=caption, channels="RGB", width=450)
    st.radio("Select the file format:",
             ["png", "jpeg", "svd"], key="file_type")
    buffer = io.BytesIO()
    if st.session_state["file_type"] != "svd":
        compressed_image = Image.fromarray(compressed_image)
        compressed_image.save(buffer, format=str(st.session_state["file_type"]).upper())
        compressed_image = buffer.getvalue()
        st.markdown(r"**Note**: SVD compression reduces the matrix representation size, not necessarily the size of PNG/JPEG files. Saving the reconstructed image as PNG or JPEG may produce files that are similar in size or even larger than the original image.")
    else:
        st.session_state["compressor"].export_svd(buffer)
        compressed_image = buffer.getvalue()
        st.markdown(
            r"**Note**: Viewing the image in SVD format is not supported by image viewers generally available, to view the file, upload it to [Image Viewer](./image_viewer/)")

    st.download_button("Download Compressed Image", file_name=f"SVD_Compression.{st.session_state['file_type']}",
                       data=compressed_image, on_click="ignore", icon=":material/download:", mime=f"image/{st.session_state['file_type']}" if st.session_state["file_type"] != "svd" else "application/octet-stream")
else:
    st.write("Please upload an image")

if st.button("Home", icon=":material/home:"):
    st.switch_page("landing_page.py")