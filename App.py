import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import plotly.graph_objects as go
import requests
import time


# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="Skin AI System", page_icon="🔬", layout="wide")

API_URL = "https://omissive-nathan-seismologic.ngrok-free.dev/generate"
API_KEY = "secret123"
headers = {"Authorization": f"Bearer {API_KEY}"}

IMAGE_SIZE = (256, 256)
CLASS_NAMES = ['acne', 'dark spots', 'wrinkles', 'pores', 'blackheades']


# ---------------------------------------------------
# LOAD MODEL WITH CACHE
# ---------------------------------------------------
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        try:
            import kagglehub
            path = kagglehub.model_download(
                "ahmedismaiil/skin-issues-model-94-accuracy/tensorFlow2/skin-condition-cnn-94acc"
            )
            model_path = f"{path}/best_model.h5"
            st.success("Model loaded from Kaggle Hub")
        except:
            model_path = 'best_model.h5'
            st.info("Loading model from local file...")

        model = keras.models.load_model(model_path)

        dummy_input = np.zeros((1, 256, 256, 3), dtype='float32')
        model.predict(dummy_input, verbose=0)

        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Ensure the model is available locally or via Kaggle Hub.")
        return None


# ---------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------
def preprocess_image(image):
    img_array = np.array(image)

    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    img_pil = Image.fromarray(img_array)
    img_pil = img_pil.resize(IMAGE_SIZE, Image.LANCZOS)
    img_resized = np.array(img_pil)

    img_normalized = img_resized.astype('float32') / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)

    return img_batch


# ---------------------------------------------------
# PREDICTION BAR CHART
# ---------------------------------------------------
def create_prediction_chart(predictions, class_names):
    fig = go.Figure(go.Bar(
        x=predictions * 100,
        y=class_names,
        orientation='h',
        marker=dict(color=predictions * 100, colorscale='Blues', showscale=False),
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


# ---------------------------------------------------
# MAIN APP
# ---------------------------------------------------
st.title(" Skin-AI Full System")

tabs = st.tabs(["🖼️ Skin Issue Detection", "💬 Skincare Assistant Chatbot"])


# ---------------------------------------------------
# TAB 1 — IMAGE CLASSIFICATION
# ---------------------------------------------------
with tabs[0]:
    st.subheader("Upload an image to detect a skin issue")

    model = load_model()
    if model is None:
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        img_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

        if img_file:
            image = Image.open(img_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("Analyze Image 🔍"):
                with st.spinner("Processing image..."):
                    processed_img = preprocess_image(image)
                    preds = model.predict(processed_img, verbose=0)[0]

                pred_idx = np.argmax(preds)
                pred_class = CLASS_NAMES[pred_idx]
                conf = preds[pred_idx] * 100

                st.success(f"Detected: **{pred_class.upper()}** ({conf:.2f}%)")

                with col2:
                    st.write("### Confidence Chart")
                    st.plotly_chart(create_prediction_chart(preds, CLASS_NAMES), use_container_width=True)


# ---------------------------------------------------
# TAB 2 — BEAUTIFUL CHATBOT UI
# ---------------------------------------------------
with tabs[1]:
    st.subheader("Chat with the AI Skincare Assistant")

    # ---------- Custom CSS ----------
    st.markdown("""
        <style>
        .chat-container {
            max-height: 550px;
            overflow-y: auto;
            padding: 15px;
            border-radius: 12px;
            background-color: #f5f7fa;
            border: 1px solid #ddd;
            margin-bottom: 10px;
        }
        .user-msg {
            background: #0d6efd;
            color: white;
            padding: 12px 15px;
            border-radius: 12px;
            margin: 8px 0;
            width: fit-content;
            max-width: 80%;
            margin-left: auto;
        }
        .bot-msg {
            background: white;
            color: #333;
            padding: 12px 15px;
            border-radius: 12px;
            margin: 8px 0;
            width: fit-content;
            max-width: 80%;
            margin-right: auto;
            border: 1px solid #eee;
        }
        .typing {
            background: #e6e6e6;
            color: #555;
            padding: 10px 15px;
            border-radius: 12px;
            width: fit-content;
            margin-right: auto;
            margin-bottom: 8px;
            font-style: italic;
        }
        </style>
    """, unsafe_allow_html=True)

    # ---------- Initialize history ----------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ---------- Chat container (only visible after first message) ----------
    if len(st.session_state.chat_history) > 0:
        chat_html = '<div class="chat-container">'

        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                chat_html += f'<div class="user-msg">{msg["text"]}</div>'
            else:
                chat_html += f'<div class="bot-msg">{msg["text"]}</div>'

        # Typing indicator if last msg is user
        if st.session_state.chat_history[-1]["role"] == "user":
            chat_html += '<div class="typing">Assistant is typing...</div>'

        chat_html += "</div>"
        st.markdown(chat_html, unsafe_allow_html=True)

    # ---------- Input box ----------
    user_input = st.chat_input("Ask something about your skin...")

    # ---------- When user sends a message ----------
    if user_input:
        st.session_state.chat_history.append({"role": "user", "text": user_input})
        st.rerun()

    # ---------- Process assistant response ----------
    if (len(st.session_state.chat_history) > 0 
        and st.session_state.chat_history[-1]["role"] == "user"):

        question = st.session_state.chat_history[-1]["text"]

        try:
            payload = {"question": question}
            response = requests.post(API_URL, headers=headers, json=payload).json()
            bot_reply = response.get("response", "Error: No response returned.")
        except Exception as e:
            bot_reply = f"API Error: {e}"

        st.session_state.chat_history.append({"role": "assistant", "text": bot_reply})
        st.rerun()



# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p><strong>Disclaimer:</strong> This system is for educational purposes only. 
        Always consult a qualified dermatologist for medical advice.</p>
        <p>Built with ❤️ by Skin-AI Team • Powered by DEPI</p>
    </div>
    """, unsafe_allow_html=True)
