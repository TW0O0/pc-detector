import os
import streamlit as st
from openai import OpenAI

# Initialize the OpenAI client
# Try different methods to get the API key
api_key = None
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
elif os.environ.get("OPENAI_API_KEY"):
    api_key = os.environ.get("OPENAI_API_KEY")
else:
    api_key = st.text_input("Enter your OpenAI API key:", type="password")
    if not api_key:
        st.warning("Please enter your OpenAI API key to continue")
        st.stop()

client = OpenAI(api_key=api_key)

# Set up the Streamlit app interface
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
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        # Convert the image to base64 for API submission
        import base64
        bytes_data = uploaded_file.getvalue()
        base64_image = base64.b64encode(bytes_data).decode("utf-8")
        
        with st.spinner("Analyzing the image for Pachyonychia Congenita indicators..."):
            try:
                # Create the prompt with PC-specific medical context
                prompt = f"""
                You are a specialized medical AI assistant focused exclusively on identifying visual indicators of Pachyonychia Congenita (PC).
                
                Analyze this {image_type.lower()} image and identify any visual indicators consistent with Pachyonychia Congenita.
                
                Focus specifically on these PC indicators:
                - Nail features: Severe thickening, discoloration, increased curvature, subungual hyperkeratosis
                - Plantar keratoderma: Painful calluses on weight-bearing areas of feet
                - Palmar keratoderma: Thickened skin on palms
                - Oral leukokeratosis: White patches on tongue or oral mucosa
                - Follicular hyperkeratosis: Small bumps around hair follicles
                - Cysts: Particularly on face, scalp, or trunk
                
                Additional patient information: {additional_info}
                
                Provide a detailed analysis with:
                1. Whether visual indicators consistent with Pachyonychia Congenita are present
                2. Specific features you've identified that support or contradict a PC diagnosis
                3. Confidence level in your assessment
                4. Which PC subtype (PC-1, PC-2, PC-3, PC-4) these features might be most consistent with, if applicable
                5. Suggested next steps for clinical confirmation (genetic testing, biopsy, etc.)
                
                IMPORTANT: Emphasize that this is an AI assessment and not a medical diagnosis. PC is genetically confirmed and requires professional evaluation.
                """
                
                # Call the OpenAI API with the image
                response = client.chat.completions.create(
                    model="gpt-4-vision-preview",  # Use the vision-capable model
                    messages=[
                        {"role": "system", "content": "You are a medical AI assistant specializing exclusively in identifying visual indicators of Pachyonychia Congenita from images."},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]}
                    ],
                    max_tokens=1200
                )
                
                # Display the results
                st.subheader("Analysis Results")
                st.write(response.choices[0].message.content)
                
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

# Footer with credit
st.markdown("---")
st.caption("This tool is for research and educational purposes only. It is not a diagnostic tool.")
