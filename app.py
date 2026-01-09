import streamlit as st
import cv2
import numpy as np
import zipfile
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="Universal Photo Finder", page_icon="🎓")
st.title("🎓 Campus Photo Finder")
st.write("The simplest way to find your photos. Upload, Scan, and Download!")

# --- STEP 1: UPLOAD PHOTOS ---
uploaded_files = st.file_uploader("1️⃣ Select Event Photos", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

# --- STEP 2: SCAN FACE ---
cam_image = st.camera_input("2️⃣ Take a 'Profile Scan'")

if uploaded_files and cam_image:
    if st.button("🚀 Start Matching"):
        # Process the scan
        scan_bytes = np.frombuffer(cam_image.getvalue(), np.uint8)
        scan_img = cv2.imdecode(scan_bytes, cv2.IMREAD_COLOR)
        
        # Create a "Face DNA" using color histograms (very fast)
        scan_hsv = cv2.cvtColor(scan_img, cv2.COLOR_BGR2HSV)
        scan_hist = cv2.calcHist([scan_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(scan_hist, scan_hist, 0, 1, cv2.NORM_MINMAX)

        matched_images = []
        progress_bar = st.progress(0)

        for i, file in enumerate(uploaded_files):
            # Read event photo
            file_bytes = np.frombuffer(file.getvalue(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Compare the color "DNA" of the photos
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_hist = cv2.calcHist([img_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(img_hist, img_hist, 0, 1, cv2.NORM_MINMAX)
            
            # Use Correlation matching
            score = cv2.compareHist(scan_hist, img_hist, cv2.HISTCMP_CORREL)
            
            # 0.70 is a good "sweet spot" for this method
            if score > 0.70:
                matched_images.append(file)
            
            progress_bar.progress((i + 1) / len(uploaded_files))

        if matched_images:
            st.success(f"✨ Found {len(matched_images)} potential matches!")
            
            # Show previews
            cols = st.columns(3)
            for idx, f in enumerate(matched_images[:6]): # Show first 6
                cols[idx % 3].image(f, use_container_width=True)

            # Create Zip
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                for f in matched_images:
                    zf.writestr(f.name, f.getvalue())
            
            st.download_button("📥 Download All Found Photos", data=zip_buffer.getvalue(), file_name="my_photos.zip")
        else:
            st.warning("No matches found. Try scanning in better light!")
