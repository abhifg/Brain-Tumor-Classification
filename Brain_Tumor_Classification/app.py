import numpy as np
import streamlit as tf
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import image

def load_model():
    model=load_model('xception_model.h5')
    base_model=None
    for layer in model.layers:
        if layer.name=='xception':
            base_model=layer
        break
    return model,base_model

def generate_gradcam(img_arr,base_model,last_conv_layer='block14_sepcov2_act'):
    grad_model=tf.keras.models.Model(
        inputs=base_model.input,
        output=[base_model.getlayer(last_conv_layer).output,base_model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs,predictions=grad_model(img_arr)
        if pred_index is None:
            pred_index=tf.argmax(predictions[0])
        class_output=predictions[:,pred_index]
    grads=tape.gradient(class_output,conv_outputs)
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


st.title("🧠 Brain MRI Classification with Grad-CAM")

uploaded_file = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_pil = Image.open(uploaded_file).convert("RGB")
    st.image(img_pil, caption="Uploaded Image", use_column_width=True)

    img_array = image.img_to_array(img_pil.resize((299, 299)))
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    with st.spinner("Classifying and generating Grad-CAM..."):
        heatmap, predictions = generate_gradcam(img_array, base_model)
        pred_class = np.argmax(predictions[0])
        pred_prob = predictions[0][pred_class]

        # You may need to replace these with your actual class names
        class_names = ["Class 0", "Class 1", "Class 2", "Class 3"]
        predicted_label = class_names[pred_class]

        st.success(f"🩺 Prediction: **{predicted_label}** with probability {pred_prob:.2f}")

        gradcam_img = overlay_heatmap(img_pil, heatmap)
        st.image(gradcam_img, caption="Grad-CAM Heatmap", use_column_width=True)
    
