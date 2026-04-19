import streamlit as st
import numpy as np
import cv2 as cv
import io
from PIL import Image
from application import image_compression

st.set_page_config(
    page_title="SVD Lab | Compressor",
    page_icon=":material/grid_4x4:",
)
st.title("Compressor", text_alignment="center")

image = st.file_uploader("Upload your image", type=["png", "jpg", "jpeg"])
st.checkbox("Greyscale", value=True, key="greyscale")

if image is not None:
    if st.session_state["greyscale"]:
        image = cv.imdecode(np.frombuffer(image.read(), dtype=np.uint8), cv.IMREAD_GRAYSCALE)
    else:
        image = cv.imdecode(np.frombuffer(image.read(), dtype=np.uint8), cv.IMREAD_COLOR)

    k = min(image.shape[:2])
    st.number_input("k", min_value=1, max_value=k, value=image_compression.get_k_from_compression_ratio(image.shape,
                                                                                                   0.25), key="k")
    compressed_image = cv.cvtColor(np.clip(image_compression.optimized_compress(image, st.session_state["k"]), 0, 255).astype("uint8"),
                        cv.COLOR_BGR2RGB)
    st.image(compressed_image, caption="Compressed Image",
             width=300)
    st.radio("Select the file format:",
             ["png", "jpeg"], key="file_type")
    compressed_image = Image.fromarray(compressed_image)
    buffer = io.BytesIO()
    compressed_image.save(buffer, format=str(st.session_state["file_type"]).upper())
    compressed_image = buffer.getvalue()

    st.markdown(r"**Note**: Downloading the image in above formats would not show difference in disk size and could be greater than original image")
    st.download_button("Download Compressed Image", file_name=f"SVD_Compression.{st.session_state['file_type']}",
                       data=compressed_image, on_click="ignore", icon=":material/download:")
else:
    st.write("Please upload an image")

if st.button("Home", icon=":material/home:"):
    st.switch_page("landing_page.py")