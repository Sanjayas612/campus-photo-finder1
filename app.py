import streamlit as st
import os
from deepface import DeepFace
import zipfile
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="Google AI Photo Finder", page_icon="🎓")
st.title("🎓 Google-Powered Photo Finder")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("⚙️ AI Settings")
# Sensitivity slider: Lower is stricter, higher is more relaxed
# For 'FaceNet' with 'cosine' distance, the default is usually around 0.40
sensitivity = st.sidebar.slider(
    "AI Sensitivity (Threshold)", 
    min_value=0.1, 
    max_value=0.8, 
    value=0.40, 
    step=0.01,
    help="Lower = Stricter (High Accuracy), Higher = Relaxed (Catch More Photos)"
)

# 1. Folder Selection (Multi-upload)
uploaded_files = st.file_uploader("1️⃣ Select Event Photos", accept_multiple_files=True, type=['jpg', 'png', 'jpeg'])

# 2. Live Face Scan
cam_image = st.camera_input("2️⃣ Scan your face to learn")

if uploaded_files and cam_image:
    if st.button("🚀 Start AI Matching"):
        img = Image.open(cam_image)
        temp_selfie = "selfie.jpg"
        img.save(temp_selfie)
        
        st.info(f"AI is searching using sensitivity: {sensitivity}...")
        
        matched_files = []
        progress_bar = st.progress(0)
        
        for i, file in enumerate(uploaded_files):
            temp_event = f"temp_{file.name}"
            with open(temp_event, "wb") as f:
                f.write(file.getbuffer())
            
            try:
                # DeepFace verify using our slider's sensitivity
                result = DeepFace.verify(
                    img1_path = temp_selfie, 
                    img2_path = temp_event, 
                    model_name = "FaceNet",
                    distance_metric = "cosine",
                    threshold = sensitivity, # <--- Slider value used here
                    enforce_detection = False
                )
                
                if result["verified"]:
                    matched_images.append(file)
            except:
                pass 
            
            if os.path.exists(temp_event):
                os.remove(temp_event)
                
            progress_bar.progress((i + 1) / len(uploaded_files))

        # --- RESULTS ---
        if matched_files:
            st.balloons()
            st.success(f"✨ Found {len(matched_files)} photos!")
            
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                for f in matched_files:
                    zf.writestr(f.name, f.getvalue())
            
            st.download_button("📥 Download Zip", data=zip_buffer.getvalue(), file_name="matches.zip")
        else:
            st.warning("No matches found. Try increasing the Sensitivity slider and scanning again!")
