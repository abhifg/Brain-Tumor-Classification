import numpy as np
import streamlit as st
import tensorflow as tf
import os
import gdown
import cv2
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

MODEL_PATH = "xception_model.h5"
MODEL_ID = "1mM9CHnWj90p8Rfi7QXyqQou4JE_CnT1d"
LAST_CONV_LAYER = "block14_sepconv2_act"

def download_model():
    url = f"https://drive.google.com/uc?id={MODEL_ID}"
    gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    else:
        st.error("❌ Model download failed!")
        raise FileNotFoundError("Model could not be downloaded.")

    return keras_load_model(MODEL_PATH)
model = download_model()
base_model = None
for layer in model.layers:
    if layer.name == 'xception':
        base_model = layer
        break
if base_model is None:
    raise ValueError("Xception not found!")
_ = model.predict(np.zeros((1, 299, 299, 3)))
def make_gradcam_heatmap(img_array, base_model, last_conv_layer_name, pred_index=None):

    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            base_model.get_layer(last_conv_layer_name).output,
            base_model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_output = predictions[:, pred_index]

    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()
def overlay_gradcam(img_pil, heatmap, alpha=0.4):
    img = np.array(img_pil.resize((299, 299)))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    superimposed_img=cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    return superimposed_img

# Streamlit UI
st.title("🧠 Brain MRI Classification with Grad-CAM")

uploaded_file = st.file_uploader("📤 Upload a brain MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    st.image(pil_img, caption="🖼️ Uploaded Image", use_column_width=True)

    img_array = image.img_to_array(pil_img.resize((299, 299)))
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    with st.spinner("🔍 Loading model & predicting..."):
        

        try:
            preds = model.predict(img_array)
            pred_class = np.argmax(preds[0])
            pred_prob = preds[0][pred_class]

            if pred_class == 0:
                statement = "🩺 REPORT: The cell shows **Tumor (Glioma)**"
            elif pred_class == 1:
                statement = "🩺 REPORT: The cell shows **Tumor (Meningioma)**"
            elif pred_class == 2:
                statement = "🩺 REPORT: The cell shows **No Tumor**"
            elif pred_class == 3:
                statement = "🩺 REPORT: The cell shows **Tumor (Pituitary)**"
            else:
                statement = f"🩺 REPORT: Unknown Class ({pred_class})"

            st.subheader(statement)

            # If tumor (pred_class 0,1,2), show heatmap
            if pred_class in [0,1,3]:
                heatmap = make_gradcam_heatmap(img_array, base_model, LAST_CONV_LAYER)
                gradcam_img = overlay_gradcam(pil_img, heatmap)
                st.image(gradcam_img, caption="🔥 Tumor Area (Grad-CAM)", use_column_width=True)

        except Exception as e:
            st.error(f"❌ Something went wrong: {e}")
# Footer
st.markdown("""
<hr style='border:1px solid #bbb'>
<div style='text-align: center; color: gray; font-size: 14px;'>
    &copy; Made by : Abhirup Ghosh - 2025
</div>
""", unsafe_allow_html=True)
