import streamlit as st

st.set_page_config(
    page_title="SVD Lab",
    page_icon=":material/grid_4x4:"
)

st.title("SVD Lab", text_alignment="center")
st.markdown(r"""
    <h2 align='center'>Explore SVD here!</h2>
""", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    with st.container(border=True, height=300, key="playground_container",horizontal=True, horizontal_alignment="center"):
        st.markdown(r"<h2 align='center'>Playground</h2>", unsafe_allow_html=True)
        st.text("Try out the image compression using singular value decomposition, without the need of providing an image and view instantly in interactive mode.",
                text_alignment="center")
        if st.button("Try Now!", key="playground"):
            st.switch_page("pages/playground.py")

    with st.container(border=True, height=305, key="notebook_container",horizontal=True, horizontal_alignment="center"):
        st.markdown("<h2 align='center'>Check out the mathematics</h2>", unsafe_allow_html=True)
        st.text("Check my notebook about the working of SVD and its application to image compression.",
                text_alignment="center")
        st.link_button(label="Take me to Notebook",
                       url="https://OjasRane.github.io/Singular-Value-Decomposition/understanding_svd_with_image_compression.html",
                       icon=":material/import_contacts:")

with col2:
    with st.container(border=True, height=300, key="compressor_container",horizontal=True, horizontal_alignment="center"):
        st.markdown(r"<h2>Compressor</h2>", unsafe_allow_html=True)
        st.text("Compress your image using singular value decomposition and download it.",
                text_alignment="center")
        if st.button("Try Now!", key="compressor"):
            st.switch_page("pages/compressor.py")
    with st.container(border=True, height=305, key="repo_container", horizontal=True, horizontal_alignment="center"):
        st.markdown(r"<h2 align='center'>GitHub Repository</h2>", unsafe_allow_html=True)
        st.text("Check out the GitHub repository for source code.", text_alignment="center")
        st.link_button(label="GitHub Repository", url="https://github.com/OjasRane/Singular-Value-Decomposition")

with st.container(border=True, height=250, key="connect_with_me", horizontal_alignment="center"):
    st.markdown("## Connect with me", text_alignment="center")
    st.image("https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white",
             link="https://github.com/OjasRane", width=180)
    st.image("https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white",
             link="https://linkedin.com/in/ojasrane-in", width=180)
