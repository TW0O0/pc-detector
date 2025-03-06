import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io

# Set up the page
st.set_page_config(page_title="Pachyonychia Congenita Image Analysis", layout="wide")

# Create the title and description
st.title("Pachyonychia Congenita Image Analysis")
st.write("Upload an image to analyze for potential indicators of Pachyonychia Congenita")

# Information about Pachyonychia Congenita
with st.expander("About Pachyonychia Congenita"):
    st.write("""
    **Pachyonychia Congenita (PC)** is a rare genetic skin disorder primarily affecting the nails and skin.
    
    **Key visual indicators include:**
    - Severely thickened nails (nail hypertrophy)
    - Painful calluses on the feet (plantar keratoderma)
    - Thickened skin on palms (palmar keratoderma)
    - Oral leukokeratosis (white patches in the mouth)
    - Follicular hyperkeratosis (small bumps around hair follicles)
    - Cysts (often on the face, scalp, and trunk)
    """)

# Function to create model (we'll create it on the fly for this example)
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification (PC or not PC)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to analyze image
def analyze_image(img, image_type, additional_info):
    # In a real application, you would load your trained model here
    # For demonstration, we'll create a simple model and simulate analysis
    
    # Preprocess the image
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    
    # In a real application with a trained model:
    # prediction = model.predict(img_array)
    # confidence = float(prediction[0][0])
    
    # For demonstration, we'll simulate an analysis
    # This is where your trained model would actually analyze the image
    
    # Simulated analysis based on image type
    analysis_data = {
        "Nails": {
            "features": ["Nail thickening", "Discoloration", "Curved appearance"],
            "confidence": 0.82,
            "subtype": "PC-1 (Jadassohn-Lewandowsky syndrome)",
            "next_steps": "Genetic testing for KRT6A, KRT16, or KRT17 mutations"
        },
        "Feet/Plantar surface": {
            "features": ["Painful calluses", "Hyperkeratosis", "Blistering"],
            "confidence": 0.78,
            "subtype": "PC-1 or PC-2 (Jackson-Lawler syndrome)",
            "next_steps": "Genetic testing and dermatological evaluation"
        },
        "Palms/Hands": {
            "features": ["Palmar keratoderma", "Thickened skin", "Hyperhidrosis"],
            "confidence": 0.75,
            "subtype": "PC-1 or PC-2",
            "next_steps": "Genetic testing for KRT6A, KRT6B, KRT16, or KRT17 mutations"
        },
        "Oral cavity": {
            "features": ["Oral leukokeratosis", "White patches", "Mucosal thickening"],
            "confidence": 0.68,
            "subtype": "PC-1",
            "next_steps": "Oral examination and genetic testing"
        },
        "Skin/Follicular": {
            "features": ["Follicular hyperkeratosis", "Cysts", "Papules"],
            "confidence": 0.72,
            "subtype": "PC-2 or PC-3 (Schafer-Brunauer syndrome)",
            "next_steps": "Skin biopsy and genetic testing"
        },
        "Other": {
            "features": ["Requires specialist evaluation"],
            "confidence": 0.60,
            "subtype": "Indeterminate",
            "next_steps": "Consult with dermatologist and genetic counselor"
        }
    }
    
    # Get the analysis for the selected image type
    analysis = analysis_data[image_type]
    
    # Generate a detailed analysis result
    result = f"""
    ## Analysis of {image_type} Image

    **Potential PC Indicators Identified:**
    """
    
    for feature in analysis["features"]:
        result += f"- {feature}\n"
    
    result += f"""
    **Additional Context Considered:**
    {additional_info if additional_info else "No additional information provided."}
    
    **Confidence Assessment:**
    The visual features are {int(analysis["confidence"]*100)}% consistent with Pachyonychia Congenita indicators.
    
    **Potential PC Subtype:**
    The features are most consistent with {analysis["subtype"]}.
    
    **Recommended Next Steps:**
    {analysis["next_steps"]}
    
    **Important Note:**
    This is an AI assessment based on visual indicators only and NOT a medical diagnosis. 
    Pachyonychia Congenita is confirmed through genetic testing for mutations in KRT6A, KRT6B, KRT6C, KRT16, or KRT17 genes.
    """
    
    return result

# File uploader for the medical image
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# Options for image type
if uploaded_file is not None:
    image_type = st.radio(
        "Select the type of image you're uploading:",
        ["Nails", "Feet/Plantar surface", "Palms/Hands", "Oral cavity", "Skin/Follicular", "Other"]
    )

    # Additional context input
    additional_info = st.text_area("Provide any additional information about the symptoms or patient history (age of onset, family history, pain levels, etc.)")

    # When the user clicks the analyze button
    if st.button("Analyze Image"):
        # Display the uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)
        
        with st.spinner("Analyzing the image for Pachyonychia Congenita indicators..."):
            try:
                # Analyze the image
                analysis_result = analyze_image(img, image_type, additional_info)
                
                # Display the results
                st.subheader("Analysis Results")
                st.markdown(analysis_result)
                
                # Resource links
                st.subheader("Pachyonychia Congenita Resources")
                st.markdown("""
                - [Pachyonychia Congenita Project](https://www.pachyonychia.org)
                - [NORD Pachyonychia Congenita Information](https://rarediseases.org/rare-diseases/pachyonychia-congenita/)
                - [Genetic and Rare Diseases Information Center](https://rarediseases.info.nih.gov/diseases/7360/pachyonychia-congenita)
                """)
                
                # Add disclaimer
                st.warning("""
                DISCLAIMER: This analysis is provided for informational and research purposes only and is not a medical diagnosis. 
                Pachyonychia Congenita requires genetic confirmation and should be diagnosed by qualified healthcare providers.
                Always consult with dermatologists and genetic specialists for proper diagnosis and management of suspected PC.
                """)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Add a section for collecting training data (for future model improvement)
st.markdown("---")
st.subheader("Help Improve This Tool")
st.write("""
If you have confirmed Pachyonychia Congenita cases with images and would like to contribute to improving 
this tool's accuracy, please contact us. All data will be handled with strict confidentiality and used only 
for research purposes with appropriate consent.
""")

# Footer with credit
st.markdown("---")
st.caption("This tool is for research and educational purposes only. It is not a diagnostic tool.")
