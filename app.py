import streamlit as st
import face_recognition
import numpy as np
import zipfile
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="Universal Photo Finder", page_icon="🎓")
st.title("🎓 Campus Photo Finder")

# --- STEP 1: UPLOAD REFERENCE FACE (Like Colab) ---
st.header("1️⃣ Your Reference Face")
reference_file = st.file_uploader("Upload a clear photo of YOUR face", type=['jpg', 'jpeg', 'png'])

# --- STEP 2: UPLOAD EVENT PHOTOS ---
st.header("2️⃣ Event Photos")
event_files = st.file_uploader("Select the photos to search through", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

if reference_file and event_files:
    # Sensitivity Slider: 0.6 is standard, 0.4 is very strict (high accuracy)
    tolerance = st.slider("AI Strictness", 0.3, 0.7, 0.5, help="Lower = More Accurate, Higher = More Matches")
    
    if st.button("🚀 Start Matching"):
        # 1. Load Reference Face
        ref_img = face_recognition.load_image_file(reference_file)
        ref_encodings = face_recognition.face_encodings(ref_img)
        
        if not ref_encodings:
            st.error("❌ No face found in your reference photo! Please upload a clearer shot.")
        else:
            user_dna = ref_encodings[0]
            matched_images = []
            progress_bar = st.progress(0)
            
            # 2. Scan Event Photos
            for i, file in enumerate(event_files):
                try:
                    # Convert uploaded file to face_recognition format
                    test_img = face_recognition.load_image_file(file)
                    test_encodings = face_recognition.face_encodings(test_img)
                    
                    for test_enc in test_encodings:
                        # Real Face Identification (The Colab way)
                        matches = face_recognition.compare_faces([user_dna], test_enc, tolerance=tolerance)
                        if True in matches:
                            matched_images.append(file)
                            break
                except Exception as e:
                    continue
                
                progress_bar.progress((i + 1) / len(event_files))

            # 3. Results
            if matched_images:
                st.balloons()
                st.success(f"✨ Found {len(matched_images)} matches!")
                
                # Show matches in a grid
                cols = st.columns(3)
                for idx, f in enumerate(matched_images[:9]):
                    cols[idx % 3].image(f, use_container_width=True)

                # Zip for download
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    for f in matched_images:
                        zf.writestr(f.name, f.getvalue())
                
                st.download_button("📥 Download Matches", data=zip_buffer.getvalue(), file_name="matches.zip")
            else:
                st.warning("No matches found. Try increasing the 'AI Strictness' slider slightly.")
