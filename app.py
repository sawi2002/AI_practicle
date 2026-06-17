import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load model
model = tf.keras.models.load_model("model.keras")

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# UI
st.set_page_config(page_title="Rock Paper Scissors Classifier")

st.title("✊✋✌ Rock Paper Scissors Classifier")
st.write("Upload an image and get prediction")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array, verbose=0)
    index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    # Output
    st.success(f"Prediction: {class_names[index]}")
    st.info(f"Confidence: {confidence * 100:.2f}%")
