import streamlit as st
import tensorflow as tf
import numpy as np

st.title("Klasifikasi Citra")
upload = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"])

# Load both models once
model_2 = tf.keras.models.load_model("./src/model/BurungMobileNetV2.h5")
model = tf.keras.models.load_model("./src/model/BurungResnet50.h5")

# Function to predict using the selected model
def predict(img_path, model_choice):
    class_names = ['Common_Kingfisher', 'Gray_Wagtail', 'Hoopoe', "House_Crow"]
    
    # Resize to the expected input size (150x150)
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))  
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0  # Normalize the image (same as training)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    
    if model_choice == "Regular CNN":
        output = model.predict(img_array)
    else:
        output = model_2.predict(img_array)
    
    score = tf.nn.softmax(output[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)  # Confidence in percentage
    return predicted_class, confidence

if st.button("Predict", type="primary"):
    if upload is not None:
        st.image(upload, caption="Uploaded Image", use_column_width=True)
        
        st.subheader("Hasil Prediksi menggunakan Regular CNN")
        with st.spinner("Loading..."):
            predicted_class_1, confidence_1 = predict(upload, "Regular CNN")
        st.write(f"Predicted Class: **{predicted_class_1}**")
        st.write(f"Confidence: **{confidence_1:.2f}%**")
        
        st.subheader("Hasil Prediksi menggunakan MobileNetV2")
        with st.spinner("Loading..."):
            predicted_class_2, confidence_2 = predict(upload, "MobileNetV2")
        st.write(f"Predicted Class: **{predicted_class_2}**")
        st.write(f"Confidence: **{confidence_2:.2f}%**")
    else:
        st.write("Please upload an image")
