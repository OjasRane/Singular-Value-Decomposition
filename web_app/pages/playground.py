import streamlit as st
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from optimized_method.optimized_svd import reconstruct
from metrics import metrics
from application.image_compression import get_k_from_compression_ratio
from application.principal_component_analysis import PCA

st.set_page_config(
    page_title="SVD Lab | Playground",
    page_icon=":material/grid_4x4:"
)
st.title("Playground", text_alignment="center")
st.pills(label="Select Playground", options=["Image Compression", "Principal Component Analysis"], key="application_choice",
         default="Image Compression", required=True, label_visibility="collapsed")

if st.session_state["application_choice"] == "Image Compression":
    @st.cache_resource
    def load_files(grayscale):
        if grayscale:
            return np.load("./web_app/assets/grayscale.npz")
        else:
            return np.load("./web_app/assets/color.npz")

    if st.checkbox("Grayscale", value=True, key="grayscale"):
        image = cv.imread(r"data/image.png", cv.IMREAD_GRAYSCALE)
    else:
        image = cv.imread(r"data/image.png")

    if image is None:
        st.error("Failed to load image")
        st.stop()

    k = min(image.shape[:2])

    if "k" not in st.session_state:
        st.session_state["k"] = k

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
            if "U" not in st.session_state or "S" not in st.session_state or "Vt" not in st.session_state:
                grayscale = load_files(grayscale=True)
                st.session_state["U"] = grayscale["U"]
                st.session_state["S"] = grayscale["S"]
                st.session_state["Vt"] = grayscale["Vt"]
            compressed_image = np.clip(reconstruct(st.session_state["U"], st.session_state["S"], st.session_state["Vt"], k=st.session_state["k"]), 0, 255).astype("uint8")
            st.image(compressed_image, caption=f"""Reconstructed Image
                                                   \nCompression Ratio={np.round(metrics.compression_ratio(image.shape, st.session_state['k']), 2)}""")
        else:
            decompositions = [
                "U_B", "S_B", "Vt_B",
                "U_G", "S_G", "Vt_G",
                "U_R", "S_R", "Vt_R"
            ]
            if any(i not in st.session_state for i in decompositions):
                color = load_files(grayscale=False)
                for i in decompositions:
                    st.session_state[i] = color[i]

            compressed_image = cv.merge(
                [
                    np.clip(reconstruct(st.session_state["U_B"], st.session_state["S_B"], st.session_state["Vt_B"], k=st.session_state["k"]), 0, 255).astype("uint8"),
                    np.clip(reconstruct(st.session_state["U_G"], st.session_state["S_G"], st.session_state["Vt_G"], k=st.session_state["k"]), 0, 255).astype("uint8"),
                    np.clip(reconstruct(st.session_state["U_R"], st.session_state["S_R"], st.session_state["Vt_R"], k=st.session_state["k"]), 0, 255).astype("uint8"),
                ]
            )
            st.image(compressed_image, caption=f"""Reconstructed Image
                                                           \nCompression Ratio={np.round(metrics.compression_ratio(image.shape, st.session_state['k']), 2)}""",
                     channels="BGR")

    with st.container(border=True, key="info_container"):
        st.markdown(f"For $k<{get_k_from_compression_ratio(image.shape, 1)}$ compression is feasible else the size increases! (True if data types are ignored)",
                    text_alignment="center")

    st.slider("k", min_value=1, max_value=k, value=k, key="k")

    kx = np.arange(1, st.session_state["k"]+1)
    st.pills("Select the plot to view it",
             ["Reconstruction Error", "Energy Retained"],
             default="Reconstruction Error",
             required=True,
             key="plot")

    if st.session_state["plot"] == "Reconstruction Error":
        error = np.empty(st.session_state["k"])
        if image.ndim == 2:
            for i in kx:
                error[i-1] = metrics.reconstruction_error(st.session_state["S"], i)
        else:
            for i in kx:
                error[i-1] = np.sqrt(metrics.reconstruction_error_squared(st.session_state["S_B"], i) +
                                     metrics.reconstruction_error_squared(st.session_state["S_G"], i) +
                                     metrics.reconstruction_error_squared(st.session_state["S_R"], i))

        st.markdown(r"""
                        $$
                        \text{Reconstruction Error}=\sqrt{\sum_{i=k}^{\min{(m,n)}}{\sigma_{i}^{2}}}
                        $$
                        """, text_alignment="center")

        fig, ax = plt.subplots()
        ax.plot(kx, error)
        ax.set(xlabel="k", ylabel="Reconstruction Error", title="Reconstruction Error")
        ax.hlines(y=error[st.session_state["k"] - 1], xmin=0, xmax=st.session_state["k"], linestyle="--", colors="black")
        ax.plot(st.session_state["k"], error[st.session_state["k"] - 1], ".", color="black")
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

    else:
        energy_retained = np.empty(st.session_state["k"])
        if image.ndim == 2:
            for i in kx:
                energy_retained[i-1] = metrics.energy_retained(st.session_state["S"], i)
        else:
            for i in kx:
                energy_retained[i-1] = (np.sum(st.session_state["S_B"][:i]**2 + st.session_state["S_G"][:i]**2 + st.session_state["S_R"][:i]**2)) / (np.sum(st.session_state["S_B"]**2 + st.session_state["S_G"]**2 + st.session_state["S_R"]**2))

        st.markdown(r"""
        $$
        \text{Energy Retained}=\frac{\sum_{i=1}^{k}{\sigma_{i}^{2}}}{\sum{\sigma_{i}^{2}}}
        $$
        """, text_alignment="center")

        fig, ax = plt.subplots()
        ax.hlines(y=energy_retained[st.session_state["k"] - 1], xmin=0, xmax=st.session_state["k"], linestyle="--", colors="black")
        ax.plot(kx, energy_retained)
        ax.set(xlabel="k", ylabel="Energy Retained", title="Energy Retained")

        ax.plot(st.session_state["k"], energy_retained[st.session_state["k"] - 1], ".", color="black")
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

elif st.session_state["application_choice"] == "Principal Component Analysis":
    st.number_input("Enter a seed number",min_value=1, value=42, key="seed")
    rng = np.random.default_rng(seed=st.session_state["seed"])
    st.number_input("Enter number of samples", min_value=10, max_value=300, value=100, key="samples")
    st.number_input("Enter scaling factor for dataset along X-axis", min_value=1.0, max_value=20.0, value=5.0, step=0.1, key="x_scaling")
    st.number_input("Enter scaling factor for dataset along Y-axis", min_value=1.0, max_value=20.0, value=2.5, step=0.1, key="y_scaling")
    st.number_input("Enter the angle in degrees to rotate from positive X-axis", min_value=0.0, max_value=360.0, value=40.0, step=0.1, key="rotate_angle")
    st.number_input("Enter number of samples for which projection lines are to be plotted", min_value=10, max_value=st.session_state["samples"], value=10, key="projection_lines")

    R = np.array([[np.cos(np.deg2rad(st.session_state["rotate_angle"])), -np.sin(np.deg2rad(st.session_state["rotate_angle"]))],
                  [np.sin(np.deg2rad(st.session_state["rotate_angle"])), np.cos(np.deg2rad(st.session_state["rotate_angle"]))]])

    X = rng.normal(size=(st.session_state["samples"], 2))
    X[:, 0] *= st.session_state["x_scaling"]
    X[:, 1] *= st.session_state["y_scaling"]
    X = X @ R.T

    needs_recalculation = (
        "X" not in st.session_state or
        "pca" not in st.session_state or
        not np.array_equal(X, st.session_state.get("X"))
    )

    st.pills("Select the plots",
             options=["Original Dataset", "Original Dataset with Principal Axes", "Data points projected on Principal Axes"],
             default="Original Dataset", required=True,
             key="plot_choice")

    if needs_recalculation:
        st.session_state["pca"] = PCA(n_components=2)
        st.session_state["projection"] = st.session_state["pca"].fit_transform(X)
        st.session_state["X"] = X

    mean = st.session_state["pca"].mean
    pc1 = st.session_state["pca"].V[:, 0]
    pc2 = st.session_state["pca"].V[:, 1]
    distance = np.max(np.linalg.norm(X - mean, axis=1))
    p = np.array([[mean - distance*pc1, mean + distance*pc1],
                  [mean - distance*pc2, mean + distance*pc2]])
    pc1_projection = st.session_state["projection"][:, 0].reshape(-1, 1) @ pc1.reshape(1, -1) + mean
    pc2_projection = st.session_state["projection"][:, 1].reshape(-1, 1) @ pc2.reshape(1, -1) + mean

    if st.session_state["plot_choice"] == "Original Dataset":
        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1], marker=".", label="Data")
        ax.scatter(mean[0], mean[1], marker=".", label="Mean", color="black")
        ax.legend(loc="upper right")
        ax.axis("equal")
        st.pyplot(fig)
        plt.close(fig)
    elif st.session_state["plot_choice"] == "Original Dataset with Principal Axes":
        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1], marker=".", label="Data")
        ax.scatter(mean[0], mean[1], marker=".", label="Mean", color="black")
        ax.plot(p[0, :, 0], p[0, :, 1], label=r"$\text{PC}1$", color="red")
        ax.plot(p[1, :, 0], p[1, :, 1], label=r"$\text{PC}2$", color="green")
        rng_projection = np.random.default_rng()
        for i in rng_projection.choice(np.arange(st.session_state["X"].shape[0]), size=st.session_state["projection_lines"], replace=False):
            ax.plot((pc1_projection[i, 0], st.session_state["X"][i, 0]), (pc1_projection[i, 1], st.session_state["X"][i, 1]),
                     color="#cacdcd", linestyle="--", alpha=0.6)
            ax.plot((pc2_projection[i, 0], st.session_state["X"][i, 0]), (pc2_projection[i, 1], st.session_state["X"][i, 1]),
                     color="#cacdcd", linestyle="--",alpha=0.6)
        ax.legend(loc="upper right")
        ax.axis("equal")
        st.pyplot(fig)
        plt.close(fig)
    elif st.session_state["plot_choice"] == "Data points projected on Principal Axes":
        fig, ax = plt.subplots()
        ax.plot(p[0, :, 0], p[0, :, 1], label=r"$\text{PC}1$", color="red")
        ax.plot(p[1, :, 0], p[1, :, 1], label=r"$\text{PC}2$", color="green")
        ax.scatter(pc1_projection[:, 0], pc1_projection[:, 1], marker=".", label=r"$\text{PC}1$", color="red")
        ax.scatter(pc2_projection[:, 0], pc2_projection[:, 1], marker=".", label=r"$\text{PC}2$", color="green")
        ax.legend(loc="upper right")
        ax.axis("equal")
        st.pyplot(fig)
        plt.close(fig)

if st.button("Home", icon=":material/home:"):
   st.switch_page(r"landing_page.py")