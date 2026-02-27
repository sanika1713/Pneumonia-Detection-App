import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load model
model = load_model('model.h5')

IMG_SIZE = 64

st.title("🫁 Pneumonia Detection")

st.write("Upload a Chest X-ray image to detect Pneumonia")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", width='stretch')

    img_array = preprocess_image(image)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.error("✅ Normal (No Pneumonia)")
    else:

        st.success("⚠️ Pneumonia Detected")
