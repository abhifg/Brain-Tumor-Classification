import numpy as np
import streamlit as st
import tensorflow as tf
import os
import cv2
import gdown
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model as keras_load_model
from PIL import Image

MODEL_PATH = "xception_model.h5"
MODEL_ID = "1mM9CHnWj90p8Rfi7QXyqQou4JE_CnT1d"

TUMOR_CLASSES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

def download_and_load_model():
    st.info("📥 Downloading model file from Google Drive (every run)...")
    url = f"https://drive.google.com/uc?id={MODEL_ID}"

    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

    gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

    # Check file size
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        st.write(f"✅ Model downloaded. Size: {size_mb:.2f} MB")
    else:
        st.error("❌ Model download failed!")
        raise FileNotFoundError("Model could not be downloaded.")

    try:
        model = keras_load_model(MODEL_PATH)
        base_model = None
        for layer in model.layers:
            if layer.name == 'xception':
                base_model = layer
                break
        if base_model is None:
            base_model = model
        return model, base_model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        raise


def generate_gradcam(img_arr, base_model, last_conv_layer='block14_sepconv2_act', pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            base_model.get_layer(last_conv_layer).output,
            base_model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_arr)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_output = predictions[:, pred_index]

    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy(), predictions


def overlay_heatmap(img_pil, heatmap, alpha=0.4):
    img = np.array(img_pil.resize((299, 299)))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return superimposed_img


# Streamlit UI
st.title("🧠 Brain MRI Classification with Grad-CAM")

uploaded_file = st.file_uploader("📤 Upload a brain MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_pil = Image.open(uploaded_file).convert("RGB")
    st.image(img_pil, caption="🖼️ Uploaded Image", use_column_width=True)

    img_array = image.img_to_array(img_pil.resize((299, 299)))
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    with st.spinner("🔍 Downloading & loading model, generating Grad-CAM..."):
        model, base_model = download_and_load_model()
        heatmap, predictions = generate_gradcam(img_array, base_model)
        pred_class = int(np.argmax(predictions[0]))
        pred_prob = float(predictions[0][pred_class])

        st.write(f"🔷 Predictions: {predictions}")
        st.write(f"🔷 Predicted class index: {pred_class}")
        st.write(f"🔷 Predicted probability: {pred_prob:.2f}")

        if pred_class < len(TUMOR_CLASSES):
            predicted_label = TUMOR_CLASSES[pred_class]
        else:
            predicted_label = f"Unknown class ({pred_class})"

        st.success(f"🩺 Prediction: **{predicted_label}** with probability {pred_prob:.2f}")

        gradcam_img = overlay_heatmap(img_pil, heatmap)
        st.image(gradcam_img, caption="🔥 Grad-CAM Heatmap", use_column_width=True)
