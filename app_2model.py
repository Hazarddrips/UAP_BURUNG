import streamlit as st
import tensorflow as tf
import numpy as np

# Set the page config for a cleaner look
st.set_page_config(page_title="Klasifikasi Citra Ternak Burung Hias", page_icon="🦜", layout="wide")

# Custom CSS for styling the app
st.markdown("""
    <style>
        .title {
            font-size: 40px;
            font-weight: bold;
            color: #525B44; /* Vibrant blue text for the title */
            text-align: center;
            text-shadow: 2px 2px 10px rgba(0, 0, 0, 0.7); /* Add shadow for better visibility */
        }
        .header {
            text-align: center;
            font-size: 24px;
            color: #ffffff;
        }
        .result {
            font-size: 20px;
            color: #4CAF50;
            font-weight: bold;
        }
        .confidence {
            font-size: 18px;
            color: #FF5722;
        }
        .spinner {
            text-align: center;
        }
        .img-container {
            margin-bottom: 20px;
            border: 2px solid #4CAF50;
            padding: 10px;
            border-radius: 10px;
        }
        .stButton > button {
            background-color: #525B44; /* Vibrant blue button */
            color: white;
            font-size: 18px;
            padding: 10px 20px;
            border-radius: 10px;
            width: 100%;
            border: none;
            transition: background-color 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #000000; /* Slightly darker blue on hover */
        }
    </style>
""", unsafe_allow_html=True)

# Page title
st.markdown('<h1 class="title">Klasifikasi Citra Ternak Burung Hias</h1>', unsafe_allow_html=True)

# Upload images section
uploads = st.file_uploader("Upload Gambar Burung Hias (jpg, png, jpeg)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# Load both models once
model_2 = tf.keras.models.load_model("./src/model/BurungMobileNetV2.h5")
model = tf.keras.models.load_model("./src/model/BurungResnet50.h5")

# Function to predict using the selected model
def predict(image_file, model_choice):
    class_names = ['Common_Kingfisher', 'Gray_Wagtail', 'Hoopoe', "House_Crow"]
    
    # Resize to the expected input size (224x224)
    img = tf.keras.utils.load_img(image_file, target_size=(224, 224))  
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0  # Normalize the image (same as training)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    
    # Model prediction
    if model_choice == "ResNet":
        output = model.predict(img_array)
    else:
        output = model_2.predict(img_array)
    
    score = tf.nn.softmax(output[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)  # Confidence in percentage
    return predicted_class, confidence

# Button to trigger predictions
if st.button("Predict", key="predict", use_container_width=True):
    if uploads:
        for upload in uploads:
            # Display the uploaded image
            st.markdown('<div class="img-container">', unsafe_allow_html=True)
            st.image(upload, caption=f"Uploaded Image: {upload.name}", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display predictions with a spinner for loading
            st.subheader(f"Hasil Prediksi untuk {upload.name} menggunakan ResNet50")
            with st.spinner("Processing with ResNet..."):
                predicted_class_1, confidence_1 = predict(upload, "ResNet")
            st.markdown(f'<p class="result">Predicted Class: {predicted_class_1}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="confidence">Confidence: {confidence_1:.2f}%</p>', unsafe_allow_html=True)
            
            st.subheader(f"Hasil Prediksi untuk {upload.name} menggunakan MobileNetV2")
            with st.spinner("Processing with MobileNetV2..."):
                predicted_class_2, confidence_2 = predict(upload, "MobileNetV2")
            st.markdown(f'<p class="result">Predicted Class: {predicted_class_2}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="confidence">Confidence: {confidence_2:.2f}%</p>', unsafe_allow_html=True)
    else:
        st.error("Silakan unggah gambar terlebih dahulu.")
