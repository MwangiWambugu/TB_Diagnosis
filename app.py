from PIL import Image
import numpy as np
import streamlit as st
import pickle

# Load model
with open("tb_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("TB X-ray Classifier")

uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Match model input size: RGB and 224x224
    image = image.resize((224, 224)).convert("RGB")
    image_array = np.array(image).flatten().reshape(1, -1)  # Shape: (1, 150528)

    prediction = model.predict(image_array)[0]

    st.write(f"Prediction: **{prediction}**")
