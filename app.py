import streamlit as st
import cv2
import numpy as np
import os
from deepface import DeepFace
import zipfile
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="Google AI Photo Finder", page_icon="🎓")
st.title("🎓 Google-Powered Photo Finder")
st.write("Using FaceNet AI to find your matches.")

# 1. Folder Selection (Multi-upload)
uploaded_files = st.file_uploader("1️⃣ Select Event Photos", accept_multiple_files=True, type=['jpg', 'png', 'jpeg'])

# 2. Live Face Scan
cam_image = st.camera_input("2️⃣ Scan your face to learn")

if uploaded_files and cam_image:
    if st.button("🚀 Start AI Matching"):
        # Save camera scan to a temp file for DeepFace to read
        img = Image.open(cam_image)
        temp_selfie = "selfie.jpg"
        img.save(temp_selfie)
        
        st.info("AI is learning your facial features...")
        
        matched_files = []
        progress_bar = st.progress(0)
        
        for i, file in enumerate(uploaded_files):
            # Save event photo temporarily
            temp_event = f"temp_{file.name}"
            with open(temp_event, "wb") as f:
                f.write(file.getbuffer())
            
            try:
                # AI Comparison using FaceNet (Google's Model)
                result = DeepFace.verify(
                    img1_path = temp_selfie, 
                    img2_path = temp_event, 
                    model_name = "FaceNet",
                    distance_metric = "cosine",
                    enforce_detection = False
                )
                
                if result["verified"]:
                    matched_files.append(file)
            except Exception as e:
                pass # Skip if no face is found in that specific photo
            
            # Cleanup temp file
            if os.path.exists(temp_event):
                os.remove(temp_event)
                
            progress_bar.progress((i + 1) / len(uploaded_files))

        if matched_files:
            st.balloons()
            st.success(f"✨ Found {len(matched_files)} photos!")
            
            # Create Zip
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                for f in matched_files:
                    zf.writestr(f.name, f.getvalue())
            
            st.download_button("📥 Download Zip", data=zip_buffer.getvalue(), file_name="matches.zip")
        else:
            st.warning("No matches found. Try scanning your face again in better light!")
