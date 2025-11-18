import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import plotly.graph_objects as go

st.set_page_config(
    page_title="Skin AI",
    page_icon="🔬",
    layout="wide"
)

IMAGE_SIZE = (256, 256)
CLASS_NAMES = ['acne', 'dark spots', 'wrinkles', 'pores', 'blackheades']

# CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    .metric-card {
        padding: 15px;
        border-radius: 8px;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Cache the model loading
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        # Try to load from kagglehub first
        try:
            import kagglehub
            path = kagglehub.model_download("ahmedismaiil/skin-issues-model-94-accuracy/tensorFlow2/skin-condition-cnn-94acc")
            model_path = f"{path}/best_model.h5"
            st.success(f"✅ Model loaded from Kaggle Hub")
        except:
            # Fallback to local file
            model_path = 'best_model.h5'
            st.info("Loading model from local file...")
        
        model = keras.models.load_model(model_path)
        
        # Warm up the model with a dummy prediction
        dummy_input = np.zeros((1, 256, 256, 3), dtype='float32')
        model.predict(dummy_input, verbose=0)
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please ensure the model is available either through Kaggle Hub or as 'best_model.h5' locally.")
        return None

def preprocess_image(image):
    """Preprocess the uploaded image for model prediction"""
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Convert to RGB if necessary (faster checks)
    if img_array.ndim == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    
    # Resize image using PIL (faster than cv2)
    from PIL import Image as PILImage
    img_pil = PILImage.fromarray(img_array)
    img_pil = img_pil.resize(IMAGE_SIZE, PILImage.LANCZOS)
    img_resized = np.array(img_pil)
    
    # Normalize pixel values
    img_normalized = img_resized.astype('float32') / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

def create_prediction_chart(predictions, class_names):
    """Create a horizontal bar chart for predictions"""
    fig = go.Figure(go.Bar(
        x=predictions * 100,
        y=class_names,
        orientation='h',
        marker=dict(
            color=predictions * 100,
            colorscale='Blues',
            showscale=False
        ),
        text=[f'{p:.1f}%' for p in predictions * 100],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Prediction Confidence (%)",
        xaxis_title="Confidence",
        yaxis_title="Skin Issue",
        height=400,
        showlegend=False,
        xaxis=dict(range=[0, 100])
    )
    
    return fig

# Main App
def main():
    # Header
    st.markdown('<p class="main-header">🔬 Skin-AI System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an image to detect skin issues using Skin-AI System</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This application uses a Custom CNN model to classify skin issues into 5 categories:
        - 🔴 **Acne**
        - 🟤 **Dark Spots**
        - 📏 **Wrinkles**
        - ⚪ **Pores**
        - ⚫ **Blackheads**
        """)
        
        st.header("📊 Model Information")
        st.write(f"""
        - **Architecture**: VGG16-based CNN
        - **Input Size**: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}
        - **Classes**: {len(CLASS_NAMES)}
        - **Accuracy**: ~95%
        """)
        
        st.header("🎯 How to Use?")
        st.write("""
        1. Upload an image of skin.
        2. Wait for the model to process.
        3. View the prediction results.
        4. Check confidence scores.
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of the skin area"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Add predict button
            predict_button = st.button("🔍 Analyze Image", type="primary", use_container_width=True)
            
        else:
            predict_button = False
            st.info("👆 Please upload an image to begin analysis")
    
    with col2:
        st.subheader("📋 Prediction Results")
        
        if uploaded_file is not None and predict_button:
            with st.spinner("🔄 Analyzing image..."):
                # Preprocess image
                processed_img = preprocess_image(image)
                
                # Make prediction
                import time
                start_time = time.time()
                predictions = model.predict(processed_img, verbose=0)[0]
                end_time = time.time()
                
                predicted_class_idx = np.argmax(predictions)
                predicted_class = CLASS_NAMES[predicted_class_idx]
                confidence = predictions[predicted_class_idx] * 100
                
                # Mark that first prediction is done
                st.session_state.first_prediction = True
                
                # Display results
                st.success(f"✅ Analysis Complete! (Took {end_time - start_time:.2f}s)")
                
                # Main prediction
                st.markdown(f"""
                <style>
                    .prediction-box {{
                        background-color: #1E1E1E;
                        padding: 20px;
                        border-radius: 15px;
                        box-shadow: 0 0 15px rgba(0,0,0,0.3);
                        margin-top: 20px;
                        color: white;  /* default text color */
                    }}
                    .prediction-box h2 {{
                        color: #FF6B6B;
                    }}
                    .prediction-box h3 {{
                        color: #4ECDC4;
                    }}
                </style>

                <div class="prediction-box">
                    <h2 style="text-align: center; margin: 0;">
                        Detected Issue: <span>{predicted_class.upper()}</span>
                    </h2>
                    <h3 style="text-align: center; margin: 10px 0 0 0;">
                        Confidence: {confidence:.2f}%
                    </h3>
                </div>
            """, unsafe_allow_html=True)
    
                st.write("")
                
                # Confidence chart
                fig = create_prediction_chart(predictions, CLASS_NAMES)
                st.plotly_chart(fig, use_container_width=True)
                
        else:
            st.info("🔍 Upload an image and click 'Analyze Image' to see results")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>⚠️ <strong>Disclaimer:</strong> This is a demonstration model for educational purposes. 
        Always consult healthcare professionals for medical advice.</p>
        <p>Built with Skin-AI Team • Powered by DEPI</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()