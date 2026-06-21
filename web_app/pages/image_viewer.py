import streamlit as st
from svd_encoder.svd import read_svd
import cv2 as cv
import numpy as np

st.set_page_config(
   page_title="SVD Lab | Image Viewer",
   page_icon=":material/grid_4x4:"
)

st.title("Image Viewer", text_alignment="center")
st.markdown("### View your images on the go", text_alignment="center")

image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "svd"])

if image is not None:
   if image.type not in ["image/png", "image/jpg", "image/jpeg"]:
      try:
         image, grayscale = read_svd(image)
      except ValueError as e:
         st.error(f"Wrong SVD file format: {e}", icon=":material/error:")
   else:
      image = np.array(cv.imdecode(np.frombuffer(image.read(), dtype=np.uint8), cv.IMREAD_COLOR), dtype=np.uint8)
      grayscale = False
   if not grayscale:
      image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
   st.image(image, width="content")
else:
   st.write("Please upload an image")

if st.button("Home", icon=":material/home:"):
   st.switch_page(r"landing_page.py")