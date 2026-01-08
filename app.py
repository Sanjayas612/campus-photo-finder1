import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import zipfile
from io import BytesIO
from PIL import Image

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

st.set_page_config(page_title="Universal Photo Finder", page_icon="🎓")
st.title("🎓 Campus Photo Finder (Fast Edition)")

# --- STEP 1: UPLOAD PHOTOS ---
uploaded_files = st.file_uploader("1️⃣ Select Event Photos", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

# --- STEP 2: SCAN FACE ---
cam_image = st.camera_input("2️⃣ Take a 'Profile Scan'")

if uploaded_files and cam_image:
    if st.button("🚀 Start Matching"):
        # Convert scan to a "feature map"
        scan_file = np.frombuffer(cam_image.getvalue(), np.uint8)
        scan_img = cv2.imdecode(scan_file, cv2.IMREAD_COLOR)
        
        # We'll use a simple histogram-based matching for the lightweight version
        scan_hsv = cv2.cvtColor(scan_img, cv2.COLOR_BGR2HSV)
        scan_hist = cv2.calcHist([scan_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(scan_hist, scan_hist, 0, 1, cv2.NORM_MINMAX)

        matched_images = []
        progress_bar = st.progress(0)

        for i, file in enumerate(uploaded_files):
            # Process each image
            file_bytes = np.frombuffer(file.getvalue(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Simple color-texture matching (Lightweight & Fast)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_hist = cv2.calcHist([img_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(img_hist, img_hist, 0, 1, cv2.NORM_MINMAX)
            
            # Compare
            score = cv2.compareHist(scan_hist, img_hist, cv2.HISTCMP_CORREL)
            
            if score > 0.75: # Threshold for matching
                matched_images.append(file)
            
            progress_bar.progress((i + 1) / len(uploaded_files))

        if matched_images:
            st.success(f"✨ Found {len(matched_images)} potential matches!")
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                for f in matched_images:
                    zf.writestr(f.name, f.getvalue())
            
            st.download_button("📥 Download Zip", data=zip_buffer.getvalue(), file_name="matches.zip")
        else:
            st.warning("No matches found. Try a different scan!")
