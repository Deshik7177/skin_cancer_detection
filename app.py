import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import tempfile
import os

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🧬",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
.upload-box {
    border: 2px dashed #4CAF50;
    padding: 30px;
    border-radius: 10px;
    text-align: center;
}
.title-text {
    font-size: 42px;
    font-weight: bold;
    color: #4CAF50;
}
.subtitle {
    font-size: 18px;
    color: #cccccc;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title-text">🧬 Skin Cancer Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">YOLOv8-powered medical image analysis</div>', unsafe_allow_html=True)
st.markdown("---")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📤 Upload a skin lesion image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    col1, col2 = st.columns(2)

    # -------- ORIGINAL IMAGE --------
    with col1:
        st.subheader("📷 Original Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

    # -------- PREDICTION --------
    with col2:
        st.subheader("🧠 Model Prediction")

        with st.spinner("Running YOLOv8 inference..."):
            # Save image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                image.save(tmp.name)
                temp_path = tmp.name

            # Run prediction
            results = model(temp_path, conf=0.4)

            # Get annotated image
            annotated = results[0].plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            st.image(annotated, use_container_width=True)

            os.remove(temp_path)

    # -------- DETAILS --------
    st.markdown("---")
    st.subheader("📊 Detection Details")

    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        for i, box in enumerate(boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            st.write(f"**Detection {i+1}:** Class `{model.names[cls]}` | Confidence `{conf:.2f}`")
    else:
        st.warning("No lesions detected.")

else:
    st.info("⬆️ Upload a skin image to start detection.")
