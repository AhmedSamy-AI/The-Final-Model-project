import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image


# ==============================
# Page Configuration
# ==============================

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠"
)

# Decision threshold for classifying a prediction as "Tumor".
# Default is 0.5. After running the notebook's threshold-analysis cell
# (Section 9), replace this with the printed "Best F1 threshold" value
# if you want to favor recall (catching more real tumors) over precision.
THRESHOLD = 0.5


# ==============================
# Load Model
# ==============================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("src/mlp_model.keras")


model = load_model()


# ==============================
# User Interface
# ==============================

st.title("🧠 Brain Tumor Detection")

st.write(
    "Upload a brain MRI image to classify it using the trained MLP model."
)

st.caption(
    "Educational project only — this is not a medical diagnostic tool "
    "and must not be used for real clinical decisions."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)


# ==============================
# Prediction
# ==============================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded MRI Image",
        width="stretch"
    )

    # Convert image to NumPy
    image = np.array(image)

    # Convert to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Resize
    image = cv2.resize(image, (64, 64))

    # Normalize
    image = image.astype("float32") / 255.0

    # Add channel dimension
    image = np.expand_dims(image, axis=-1)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    if st.button("Predict"):

        prediction = model.predict(image, verbose=0)[0][0]

        st.write(f"Prediction Score: {prediction:.4f}")
        st.progress(float(prediction))

        if prediction >= THRESHOLD:

            st.error("⚠️ Tumor Detected")

        else:

            st.success("✅ No Tumor Detected")