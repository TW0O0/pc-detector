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

# Model configuration
MODEL_PATH = os.path.join('model', 'saved_models', 'efficientnet_pachyonychia_final.h5')
MODEL_VERSION = "1.0"  # Add your model version here

# Load the trained model
@st.cache_resource  # Cache the model to avoid reloading
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found. Please ensure the model is available at {MODEL_PATH}")
            return None
        
        # Load model with optimizations for inference
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        
        # Optimize the model for inference
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Image preprocessing function
def preprocess_image(image, target_size=(224, 224)):
    # Resize the image
    image = image.resize(target_size)
    # Convert to array and normalize
    image_array = np.array(image) / 255.0
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Image manipulation section with cropping only
def image_manipulation_section():
    """Function that adds image cropping capability to the Streamlit app."""
    # Only show manipulation tools if an image is uploaded
    if st.session_state.image is None:
        return None
    
    image = st.session_state.image
    
    st.subheader("Image Cropping")
    st.write("Click and drag on the image to select an area to crop:")
    
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
    if st.button("Reset Image"):
        image = st.session_state.original_image.copy()
        st.session_state.crop_coords = []
        st.info("Image reset to original")
    
    # Display the manipulated image
    st.image(image, caption="Current Image", use_column_width=True)
    
    # Update the session state with the manipulated image
    st.session_state.image = image
    
    return image

# Main application flow
def main():
    # Load model at startup
    model = load_model()
    
    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        
        # Store the original image in session state for reset functionality
        if st.session_state.original_image is None or uploaded_file != st.session_state.get('last_uploaded_file'):
            st.session_state.original_image = image.copy()
            st.session_state.image = image.copy()
            st.session_state.last_uploaded_file = uploaded_file
        
        # Display the original uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Add the image cropping section
        manipulated_image = image_manipulation_section()
        
        # Use the manipulated image for analysis if it exists
        analysis_image = manipulated_image if manipulated_image is not None else image
        
        # Make prediction when the user clicks the button
        if st.button("Analyze Image"):
            if model is not None:
                # Show a spinner while processing
                with st.spinner('Analyzing image...'):
                    try:
                        # Preprocess the image
                        processed_img = preprocess_image(analysis_image)
                        
                        # Make prediction
                        prediction = model.predict(processed_img, verbose=0)[0][0]
                        
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
                    except Exception as e:
                        st.error(f"Error during image analysis: {str(e)}")
            else:
                st.error("Model could not be loaded. Please check the model file location.")

    # Add information about the research project
    st.sidebar.title("About")
    st.sidebar.info("""
    ## Pachyonychia Congenita Detector
    This is a research project aimed at developing AI tools for detecting Pachyonychia Congenita from images.
    The model uses transfer learning with EfficientNet architecture.
    For more information or to contribute to this research, please contact kushalreddywork@gmail.com.
    """)

    # Add model information
    if model is not None:
        st.sidebar.subheader("Model Information")
        st.sidebar.text(f"Architecture: EfficientNetB3")
        st.sidebar.text(f"Input size: 224x224 pixels")
        st.sidebar.text(f"Model Version: {MODEL_VERSION}")
        st.sidebar.text(f"TensorFlow Version: {tf.__version__}")

if __name__ == "__main__":
    main()
