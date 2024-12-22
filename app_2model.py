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
            color: #ffffff; /* White text for better contrast */
            text-align: center;
            text-shadow: 2px 2px 10px rgba(0, 0, 0, 0.7); /* Add shadow for better visibility */
        }
        .header {
            text-align: center;
            font-size: 24px;
            color: #333;
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
        .button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
            border-radius: 10px;
            width: 100%;
        }
        .button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

# Page title
st.markdown('<h1 class="title">Klasifikasi Citra Ternak Burung Hias</h1>', unsafe_allow_html=True)

# Upload image section
upload = st.file_uploader("Upload Gambar Burung Hias (jpg, png, jpeg)", type=["jpg", "png", "jpeg"])

# Load both models once
model_2 = tf.keras.models.load_model("./src/model/BurungMobileNetV2.h5")
model = tf.keras.models.load_model("./src/model/BurungResnet50.h5")

# Function to predict using the selected model
def predict(img_path, model_choice):
    class_names = ['Common_Kingfisher', 'Gray_Wagtail', 'Hoopoe', "House_Crow"]
    
    # Resize to the expected input size (224x224)
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))  
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
    if upload is not None:
        # Display the uploaded image
        st.markdown('<div class="img-container">', unsafe_allow_html=True)
        st.image(upload, caption="Uploaded Image", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display predictions with a spinner for loading
        st.subheader("Hasil Prediksi menggunakan ResNet50")
        with st.spinner("Processing with ResNet..."):
            predicted_class_1, confidence_1 = predict(upload, "ResNet")
        st.markdown(f'<p class="result">Predicted Class: **{predicted_class_1}**</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="confidence">Confidence: **{confidence_1:.2f}%**</p>', unsafe_allow_html=True)
        
        st.subheader("Hasil Prediksi menggunakan MobileNetV2")
        with st.spinner("Processing with MobileNetV2..."):
            predicted_class_2, confidence_2 = predict(upload, "MobileNetV2")
        st.markdown(f'<p class="result">Predicted Class: **{predicted_class_2}**</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="confidence">Confidence: **{confidence_2:.2f}%**</p>', unsafe_allow_html=True)
    else:
        st.error("Silakan unggah gambar terlebih dahulu.")
