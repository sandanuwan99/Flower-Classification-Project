import os
import numpy as np
import streamlit as st
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from io import BytesIO

# Load the model
model = load_model('Flower_Classification.keras')
flowers_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Function to predict images
def predict_images(uploaded_file):
    try:
        
        # Convert the file buffer to an image that TensorFlow can handle
        img = image.load_img(uploaded_file, target_size=(180, 180))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict the class of the image
        predictions = model.predict(img_array)
        result = tf.nn.softmax(predictions[0])
        flower_name = flowers_names[np.argmax(result)]

        return f"The image belongs to {flower_name} with a confidence of {np.max(result) * 100:.2f}%."
    except Exception as e:
        return f"Error: {e}"

# Streamlit app
st.title('Flower Classification CNN Model')

uploaded_file = st.file_uploader('Upload an Image', type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Use BytesIO to handle the uploaded file
        img_bytes = uploaded_file.read()
        st.image(BytesIO(img_bytes), caption='Uploaded Image', use_column_width=200)

        # Predict and display the result
        prediction = predict_images(BytesIO(img_bytes))
        st.success(prediction)
    except Exception as e:
        st.error(f"Error: {e}")
