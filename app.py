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
st.write("Using Google's **FaceNet** AI to find your matches in the event folder.")

# 1. Folder Selection (Multi-upload)
st.info("💡 Tip: Select all photos in your folder (Ctrl+A) and drag them here.")
uploaded_files = st.file_uploader("1️⃣ Select Event Photos", accept_multiple_files=True, type=['jpg', 'png', 'jpeg'])

# 2. Live Face Scan
cam_image = st.camera_input("2️⃣ Scan your face to learn your features")

if uploaded_files and cam_image:
    # Add a sensitivity slider for better user control
    tolerance = st.slider("AI Sensitivity", 0.1, 0.8, 0.4, help="Lower is stricter, higher catches more photos.")
    
    if st.button("🚀 Start AI Matching"):
        # Save camera scan to a temp file
        img = Image.open(cam_image)
        temp_selfie = "selfie.jpg"
        img.save(temp_selfie)
        
        st.info("🔍 AI is analyzing your face and searching the folder...")
        
        matched_files = []
        progress_bar = st.progress(0)
        
        for i, file in enumerate(uploaded_files):
            # Save event photo temporarily for the AI to read
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
                    enforce_detection = False,
                    threshold = tolerance
                )
                
                if result["verified"]:
                    matched_files.append(file)
            except Exception as e:
                continue # Skip photos where a face cannot be detected
            
            # Cleanup temp file to save server space
            if os.path.exists(temp_event):
                os.remove(temp_event)
                
            progress_bar.progress((i + 1) / len(uploaded_files))

        if matched_files:
            st.balloons()
            st.success(f"✨ Found {len(matched_files)} photos!")
            
            # Preview the first few matches
            cols = st.columns(3)
            for idx, f in enumerate(matched_files[:6]):
                cols[idx % 3].image(f, use_container_width=True)
            
            # Create Zip for download
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                for f in matched_files:
                    zf.writestr(f.name, f.getvalue())
            
            st.download_button("📥 Download All Matches (.zip)", data=zip_buffer.getvalue(), file_name="my_matches.zip")
        else:
            st.warning("No matches found. Try adjusting the sensitivity slider or scanning in better light!")
