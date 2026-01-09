import streamlit as st
import cv2
import numpy as np
import zipfile
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="Universal Photo Finder", page_icon="🎓")
st.title("🎓 Campus Photo Finder")

# --- STEP 1: UPLOAD REFERENCE ---
st.header("1️⃣ Your Reference Photo")
ref_file = st.file_uploader("Upload a clear photo of yourself", type=['jpg', 'jpeg', 'png'])

# --- STEP 2: UPLOAD EVENT FOLDER ---
st.header("2️⃣ Event Photos")
event_files = st.file_uploader("Upload the photos to search", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

if ref_file and event_files:
    # Sensitivity: Lower is stricter
    threshold = st.slider("Matching Strictness", 0.3, 0.9, 0.6, help="Higher = More Precise, Lower = More Matches")
    
    if st.button("🚀 Start Matching"):
        # Load reference and convert to grayscale for feature detection
        ref_bytes = np.frombuffer(ref_file.read(), np.uint8)
        ref_img = cv2.imdecode(ref_bytes, cv2.IMREAD_GRAYSCALE)
        
        # Detect face in reference to create a "Target Template"
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(ref_img, 1.1, 4)
        
        if len(faces) == 0:
            st.error("❌ AI couldn't find a face in your reference photo. Try a clearer shot!")
        else:
            # Get the first face found as the template
            (x, y, w, h) = faces[0]
            face_template = ref_img[y:y+h, x:x+w]
            
            matched_images = []
            progress_bar = st.progress(0)
            
            for i, file in enumerate(event_files):
                # Read event photo
                file_bytes = np.frombuffer(file.getvalue(), np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    # Template Matching: Searching for your face pattern in the large photo
                    res = cv2.matchTemplate(img, face_template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    
                    if max_val >= threshold:
                        matched_images.append(file)
                
                progress_bar.progress((i + 1) / len(event_files))

            if matched_images:
                st.balloons()
                st.success(f"✨ Found {len(matched_images)} matches!")
                
                # Show Previews
                cols = st.columns(3)
                for idx, f in enumerate(matched_images[:6]):
                    cols[idx % 3].image(f, use_container_width=True)

                # Create Zip
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    for f in matched_images:
                        zf.writestr(f.name, f.getvalue())
                
                st.download_button("📥 Download Matches", data=zip_buffer.getvalue(), file_name="matches.zip")
            else:
                st.warning("No matches found. Try lowering the 'Matching Strictness' slider.")
