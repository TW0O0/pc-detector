import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# Set page title and configuration
st.set_page_config(page_title="Pachyonychia Congenita Detector", layout="wide")

# App title and description
st.title("Pachyonychia Congenita Detector")
st.markdown("""
This application uses a deep learning model to detect signs of Pachyonychia Congenita from images.
Upload a clear image of the affected area for analysis.
""")

# Load the trained model
@st.cache_resource  # Cache the model to avoid reloading
def load_model():
    model_path = os.path.join('model', 'saved_models', 'efficientnet_pachyonychia_final.h5')
    if not os.path.exists(model_path):
        st.warning(f"Model file not found at {model_path}. Please train the model first.")
        return None
    
    return tf.keras.models.load_model(model_path)

model = load_model()

# Image preprocessing function
def preprocess_image(image, target_size=(224, 224)):
    # Resize the image
    image = image.resize(target_size)
    # Convert to array and normalize
    image_array = np.array(image) / 255.0
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Make prediction when the user clicks the button
    if st.button("Analyze Image"):
        if model is not None:
            # Show a spinner while processing
            with st.spinner('Analyzing image...'):
                # Preprocess the image
                processed_img = preprocess_image(image)
                
                # Make prediction
                prediction = model.predict(processed_img)[0][0]
                
                # Display results
                st.subheader("Analysis Results")
                
                # Create a progress bar for visualization
                if prediction >= 0.5:
                    probability = prediction * 100
                    st.error(f"Potential Pachyonychia Congenita detected with {probability:.1f}% confidence")
                else:
                    probability = (1 - prediction) * 100
                    st.success(f"No signs of Pachyonychia Congenita detected ({probability:.1f}% confidence)")
                
                # Show the probability as a progress bar
                st.progress(float(prediction))
                
                # Display disclaimer
                st.warning("Disclaimer: This tool is for research purposes only and should not replace professional medical diagnosis.")
        else:
            st.error("Model not loaded. Please train the model first.")

# Add information about the research project
st.sidebar.title("About")
st.sidebar.info("""
## Pachyonychia Congenita Detector

This is a research project aimed at developing AI tools for detecting Pachyonychia Congenita from images.

The model uses transfer learning with EfficientNet architecture and was trained on multiple datasets.

For more information or to contribute to this research, please contact kushalreddywork@gmail.com.
""")

# Add model information
if model is not None:
    st.sidebar.subheader("Model Information")
    st.sidebar.text("Architecture: EfficientNetB3")
    st.sidebar.text("Input size: 224x224 pixels")
