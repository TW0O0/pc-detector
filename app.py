import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from streamlit_image_coordinates import streamlit_image_coordinates

# Set page title and configuration
st.set_page_config(page_title="Pachyonychia Congenita Detector", layout="wide")

# App title and description
st.title("Pachyonychia Congenita Detector")
st.markdown("""
This application uses a deep learning model to detect signs of Pachyonychia Congenita from images.
Upload a clear image of the affected area for analysis.
""")

# Initialize session state for image handling
if 'image' not in st.session_state:
    st.session_state.image = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'crop_coords' not in st.session_state:
    st.session_state.crop_coords = []

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

# Image manipulation section
def image_manipulation_section():
    """Function that adds image manipulation capabilities to the Streamlit app."""
    # Only show manipulation tools if an image is uploaded
    if st.session_state.image is None:
        return None
    
    image = st.session_state.image
    
    st.subheader("Image Manipulation")
    st.write("Adjust your image before analysis:")
    
    # Create three columns for different manipulation options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Rotation
        rotation_angle = st.slider("Rotate", -180, 180, 0, 5)
        if rotation_angle != 0:
            image = image.rotate(rotation_angle, expand=True, fillcolor=(255, 255, 255))
    
    with col2:
        # Scaling
        scale_factor = st.slider("Scale", 0.5, 2.0, 1.0, 0.1)
        if scale_factor != 1.0:
            new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
            image = image.resize(new_size, Image.LANCZOS)
    
    with col3:
        # Brightness and contrast
        brightness = st.slider("Brightness", 0.5, 1.5, 1.0, 0.1)
        contrast = st.slider("Contrast", 0.5, 1.5, 1.0, 0.1)
        if brightness != 1.0 or contrast != 1.0:
            from PIL import ImageEnhance
            if brightness != 1.0:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(brightness)
            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(contrast)
    
    # Cropping section
    st.write("Click and drag on the image to crop:")
    
    # Display image for cropping with coordinate capture
    coords = streamlit_image_coordinates(image, key="crop_tool")
    
    # Handle cropping based on click coordinates
    if coords is not None and coords['x'] is not None:
        if len(st.session_state.crop_coords) < 2:
            st.session_state.crop_coords.append((coords['x'], coords['y']))
            if len(st.session_state.crop_coords) == 1:
                st.write(f"First corner selected at {st.session_state.crop_coords[0]}")
            elif len(st.session_state.crop_coords) == 2:
                st.write(f"Second corner selected at {st.session_state.crop_coords[1]}")
                st.write("Ready to crop! Click 'Apply Crop' to continue.")
    
    # Apply crop button
    if len(st.session_state.crop_coords) == 2 and st.button("Apply Crop"):
        x1, y1 = st.session_state.crop_coords[0]
        x2, y2 = st.session_state.crop_coords[1]
        # Ensure proper coordinate order (top-left, bottom-right)
        left = min(x1, x2)
        top = min(y1, y2)
        right = max(x1, x2)
        bottom = max(y1, y2)
        # Apply the crop
        image = image.crop((left, top, right, bottom))
        # Reset crop coordinates
        st.session_state.crop_coords = []
        st.success("Image cropped successfully!")
    
    # Reset crop button
    if len(st.session_state.crop_coords) > 0 and st.button("Reset Crop Selection"):
        st.session_state.crop_coords = []
        st.info("Crop selection reset")
    
    # Reset all manipulations button
    if st.button("Reset All Manipulations"):
        image = st.session_state.original_image.copy()
        st.session_state.crop_coords = []
        st.info("All manipulations reset")
    
    # Display the manipulated image
    st.image(image, caption="Manipulated Image", use_column_width=True)
    
    # Update the session state with the manipulated image
    st.session_state.image = image
    
    return image

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    
    # Store the original image in session state for reset functionality
    if st.session_state.original_image is None or uploaded_file != st.session_state.last_uploaded_file:
        st.session_state.original_image = image.copy()
        st.session_state.image = image.copy()
        st.session_state.last_uploaded_file = uploaded_file
    
    # Display the original uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Add the image manipulation section
    manipulated_image = image_manipulation_section()
    
    # Use the manipulated image for analysis if it exists
    analysis_image = manipulated_image if manipulated_image is not None else image
    
    # Make prediction when the user clicks the button
    if st.button("Analyze Image"):
        if model is not None:
            # Show a spinner while processing
            with st.spinner('Analyzing image...'):
                # Preprocess the image
                processed_img = preprocess_image(analysis_image)
                
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
