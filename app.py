
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

st.set_page_config(page_title="Deer vs Antelope Classifier", page_icon="🦌", layout="centered")

MODEL_PATH = "deer_antelope_model.keras"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = (224, 224)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
    return model, class_names

def preprocess_image(image: Image.Image):
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = tf.keras.utils.img_to_array(image)
    arr = np.expand_dims(arr, axis=0)
    return arr

def main():
    st.title("🦌 Deer vs Antelope Classifier")
    st.write(
        "Upload an image and this CNN (MobileNetV2 transfer-learning model) "
        "will predict whether it shows a **deer** or an **antelope**."
    )
    st.caption("GET 324 — Cloud Computing and AI Model Deployment for Engineering Applications")

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file `{MODEL_PATH}` not found. Place it in the same folder as this app.")
        return

    model, class_names = load_model()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Classifying..."):
            arr = preprocess_image(image)
            prob = float(model.predict(arr, verbose=0).flatten()[0])
            pred_idx = int(prob >= 0.5)
            pred_label = class_names[pred_idx]
            confidence = prob if pred_idx == 1 else 1 - prob

        st.subheader("Prediction")
        st.success(f"**{pred_label.upper()}**  (confidence: {confidence*100:.2f}%)")

        st.write("Class probabilities:")
        st.progress(float(prob))
        st.write(f"- {class_names[0]}: {(1-prob)*100:.2f}%")
        st.write(f"- {class_names[1]}: {prob*100:.2f}%")

    st.markdown("---")
    st.caption("Model: MobileNetV2 transfer learning, fine-tuned on real Kaggle image data.")

if __name__ == "__main__":
    main()
