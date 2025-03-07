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
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'custom_model_path' not in st.session_state:
    st.session_state.custom_model_path = None

# Model configuration - with multiple potential paths
DEFAULT_MODEL_PATHS = [
    os.path.join('model', 'saved_models', 'efficientnet_pachyonychia_final.h5'),
    os.path.join('models', 'efficientnet_pachyonychia_final.h5'),
    os.path.join('.', 'efficientnet_pachyonychia_final.h5')
]
MODEL_VERSION = "1.0"

# Load the trained model
@st.cache_resource  # Cache the model to avoid reloading
def load_model(model_path=None):
    """Load the model from the specified path or try default paths"""
    paths_to_try = []
    
    # Add custom path if provided
    if model_path is not None:
        paths_to_try.append(model_path)
    
    # Add default paths
    paths_to_try.extend(DEFAULT_MODEL_PATHS)
    
    # Try each path until a model is successfully loaded
    for path in paths_to_try:
        try:
            if os.path.exists(path):
                st.info(f"Loading model from: {path}")
                
                # Load model with optimizations for inference
                model = tf.keras.models.load_model(path, compile=False)
                
                # Optimize the model for inference
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                
                st.session_state.model_loaded = True
                return model
        except Exception as e:
            continue  # Try the next path
    
    # If we get here, no model was loaded
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

# Model upload and management
def model_management_section():
    """Function to allow users to upload a custom model file"""
    st.sidebar.subheader("Model Management")
    
    # Option to upload a custom model
    uploaded_model = st.sidebar.file_uploader("Upload model (.h5 file)", type=['h5'])
    
    if uploaded_model is not None:
        # Save the uploaded model to a temporary file
        model_path = os.path.join(os.getcwd(), "temp_model.h5")
        with open(model_path, "wb") as f:
            f.write(uploaded_model.getbuffer())
        
        st.sidebar.success(f"Model uploaded successfully to {model_path}")
        st.session_state.custom_model_path = model_path
        
        # Add a button to load the uploaded model
        if st.sidebar.button("Load Uploaded Model"):
            return load_model(model_path)
    
    # Add a button to search for model in default locations
    if st.sidebar.button("Search for Model in Default Locations"):
        return load_model()
    
    return None

# Main application flow
def main():
    # Add the model management section in the sidebar
    custom_model = model_management_section()
    
    # Load model - first try the custom model from the sidebar if it exists
    if custom_model is not None:
        model = custom_model
    # Otherwise try to load from default locations
    elif not st.session_state.model_loaded:
        model = load_model()
        
        # Display model loading error and instructions if needed
        if model is None:
            st.error("""
            **Model file not found!**
            
            Please try one of the following solutions:
            
            1. Upload your model file using the 'Upload model' option in the sidebar
            2. Place your model in one of these locations:
               - model/saved_models/efficientnet_pachyonychia_final.h5
               - models/efficientnet_pachyonychia_final.h5
               - ./efficientnet_pachyonychia_final.h5 (in the same directory as this app)
            3. Click 'Search for Model in Default Locations' in the sidebar to try again
            """)
    
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
            if st.session_state.model_loaded:
                # Get the model
                model = load_model(st.session_state.custom_model_path)
                
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
                st.error("Please load a model first using the sidebar options.")

    # Add information about the research project
    st.sidebar.title("About")
    st.sidebar.info("""
    ## Pachyonychia Congenita Detector
    This is a research project aimed at developing AI tools for detecting Pachyonychia Congenita from images.
    The model uses transfer learning with EfficientNet architecture.
    For more information or to contribute to this research, please contact kushalreddywork@gmail.com.
    """)

    # Add model information
    if st.session_state.model_loaded:
        model = load_model(st.session_state.custom_model_path)
        st.sidebar.subheader("Model Information")
        st.sidebar.text(f"Architecture: EfficientNetB3")
        st.sidebar.text(f"Input size: 224x224 pixels")
        st.sidebar.text(f"Model Version: {MODEL_VERSION}")
        st.sidebar.text(f"TensorFlow Version: {tf.__version__}")

if __name__ == "__main__":
    main()
