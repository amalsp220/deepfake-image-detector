"""Streamlit Web Application for Deepfake Image Detection.

This is the main UI file that provides a simple interface for users to upload
images and detect if they are deepfakes.
"""

import streamlit as st
import sys
import os
from PIL import Image
import torch
import time

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vit_detector import create_detector
from utils.preprocess import preprocess_for_detection


# Page configuration
st.set_page_config(
    page_title="Deepfake Image Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model(model_name="vit_base_patch16_224", weights_path=None):
    """Load and cache the deepfake detection model.
    
    Args:
        model_name: Name of the ViT model
        weights_path: Optional path to fine-tuned weights
        
    Returns:
        Loaded detector model
    """
    with st.spinner("Loading model... This may take a moment."):
        detector = create_detector(
            model_name=model_name,
            weights_path=weights_path
        )
    return detector


def display_prediction(prob_fake, label, threshold=0.5):
    """Display prediction results with styled output.
    
    Args:
        prob_fake: Probability that the image is fake
        label: Classification label (Real/Fake)
        threshold: Threshold for fake classification
    """
    # Color coding based on result
    if label == "Fake":
        color = "red"
        icon = "⚠️"
    else:
        color = "green"
        icon = "✅"
    
    # Display result
    st.markdown(f"### {icon} Prediction: **:{color}[{label}]**")
    
    # Display probability
    st.metric(
        label="Fake Probability",
        value=f"{prob_fake:.2%}",
        delta=f"{prob_fake - threshold:.2%} from threshold"
    )
    
    # Progress bar for visualization
    st.progress(prob_fake)
    
    # Confidence interpretation
    if prob_fake > 0.8 or prob_fake < 0.2:
        confidence = "High"
    elif prob_fake > 0.6 or prob_fake < 0.4:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    st.info(f"**Confidence Level:** {confidence}")


def main():
    """Main application function."""
    
    # Title and description
    st.title("🔍 Deepfake Image Detector")
    st.markdown("""
    Upload an image to detect if it has been manipulated using deepfake technology.
    This app uses a Vision Transformer (ViT) model to analyze images for signs of manipulation.
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        model_name = st.selectbox(
            "Model Architecture",
            ["vit_base_patch16_224", "vit_small_patch16_224", "vit_large_patch16_224"],
            index=0,
            help="Select the ViT model variant. Larger models are more accurate but slower."
        )
        
        # Threshold slider
        threshold = st.slider(
            "Detection Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Threshold for classifying an image as fake. Lower values are more sensitive."
        )
        
        # Face detection option
        detect_face = st.checkbox(
            "Enable Face Detection",
            value=False,
            help="Automatically detect and crop faces before analysis (requires MTCNN)."
        )
        
        # Device selection
        device_option = st.radio(
            "Compute Device",
            ["Auto", "CPU", "GPU"],
            index=0,
            help="Select which device to use for inference."
        )
        
        device = None if device_option == "Auto" else device_option.lower()
        
        # About section
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This application uses state-of-the-art Vision Transformer models
        to detect deepfake images.
        
        **Note:** This is a demonstration tool. Results should not be used
        as the sole basis for critical decisions.
        """)
        
        # System info
        cuda_available = torch.cuda.is_available()
        st.markdown(f"**CUDA Available:** {'Yes ✅' if cuda_available else 'No ❌'}")
        
        if cuda_available:
            st.markdown(f"**GPU:** {torch.cuda.get_device_name(0)}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Upload an image in JPG, PNG, or BMP format."
        )
        
        if uploaded_file is not None:
            # Load and display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.caption(f"Size: {image.size[0]}x{image.size[1]} | Format: {image.format}")
    
    with col2:
        st.header("📊 Analysis Results")
        
        if uploaded_file is not None:
            # Analyze button
            if st.button("🔎 Analyze Image", type="primary", use_container_width=True):
                try:
                    # Load model
                    detector = load_model(model_name, weights_path=None)
                    
                    # Preprocess image
                    with st.spinner("Preprocessing image..."):
                        processed_image = preprocess_for_detection(
                            image,
                            detect_face=detect_face,
                            resize=True
                        )
                    
                    # Run prediction
                    with st.spinner("Analyzing image..."):
                        start_time = time.time()
                        prob_fake, label = detector.predict(processed_image)
                        inference_time = time.time() - start_time
                    
                    # Adjust label based on threshold
                    label = "Fake" if prob_fake > threshold else "Real"
                    
                    # Display results
                    display_prediction(prob_fake, label, threshold)
                    
                    # Performance metrics
                    st.markdown("---")
                    st.caption(f"⏱️ Inference time: {inference_time:.3f} seconds")
                    
                    # Success message
                    st.success("Analysis complete!")
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.exception(e)
        else:
            st.info("👆 Upload an image to begin analysis.")
    
    # Tips and information
    with st.expander("💡 Tips for Best Results"):
        st.markdown("""
        - **Image Quality**: Use high-resolution images for better accuracy
        - **Face Detection**: Enable face detection for portrait images
        - **Lighting**: Images with good lighting produce better results
        - **File Format**: JPG and PNG formats work best
        - **Threshold**: Adjust the threshold based on your use case:
          - Lower threshold: More sensitive, may have false positives
          - Higher threshold: Less sensitive, may miss some fakes
        """)
    
    with st.expander("⚠️ Disclaimer"):
        st.markdown("""
        This deepfake detection tool is provided for educational and research purposes.
        
        **Important Notes:**
        - Detection accuracy depends on the quality of the image and the sophistication of the manipulation
        - This tool should not be used as the sole method for verifying image authenticity
        - False positives and false negatives can occur
        - Always verify important findings through multiple sources
        - The model may not detect all types of image manipulations
        """)


if __name__ == "__main__":
    main()
