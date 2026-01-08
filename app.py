import streamlit as st
import face_recognition
import numpy as np
import zipfile
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="Universal Photo Finder", page_icon="🎓")
st.title("🎓 Universal Campus Photo Finder")

# --- STEP 1: UPLOAD (SELECT ALL FILES IN FOLDER) ---
st.header("1️⃣ Select Event Photos")
st.write("Open your folder, press 'Ctrl+A' (or Cmd+A) to select all images, and upload.")
uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

# --- STEP 2: FACE SCAN & LEARNING ---
st.header("2️⃣ Face Scan & Learning")
st.info("Look directly at the camera. The AI will learn your unique facial features.")
cam_image = st.camera_input("Scan your face")

if uploaded_files and cam_image:
    if st.button("🚀 Start Finding Me"):
        # LEARN: Convert camera scan to facial 'DNA'
        scan_file = face_recognition.load_image_file(cam_image)
        scan_encodings = face_recognition.face_encodings(scan_file)
        
        if not scan_encodings:
            st.error("AI couldn't see your face clearly. Please adjust your lighting and scan again.")
        else:
            user_dna = scan_encodings[0]
            st.success("✅ Face learned successfully! Searching event photos...")
            
            matched_images = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            # SEARCH: Compare learned DNA against uploaded folder
            for i, file in enumerate(uploaded_files):
                # Load image from the uploaded list
                img = face_recognition.load_image_file(file)
                
                # Find all faces in this specific photo
                current_face_encodings = face_recognition.face_encodings(img)
                
                # Check if ANY of the faces in the photo match your learned DNA
                for enc in current_face_encodings:
                    # 0.53 is our 'Sweet Spot' tolerance
                    match = face_recognition.compare_faces([user_dna], enc, tolerance=0.53)
                    if match[0]:
                        matched_images.append(file)
                        break # Found you in this photo, move to next photo
                
                progress_bar.progress((i + 1) / len(uploaded_files))
                status_text.text(f"Processed {i+1} of {len(uploaded_files)} photos...")

            # --- STEP 3: RESULTS & ZIP ---
            if matched_images:
                st.balloons()
                st.success(f"✨ Found {len(matched_images)} photos of you!")
                
                # Show matches in a nice grid
                cols = st.columns(3)
                for idx, f in enumerate(matched_images):
                    cols[idx % 3].image(f, use_container_width=True)

                # Create the Zip file for download
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    for f in matched_images:
                        zf.writestr(f.name, f.getvalue())
                
                st.download_button("📥 Download My Photos (.zip)", data=zip_buffer.getvalue(), file_name="my_photos.zip")
            else:
                st.warning("No photos found. Try a different face scan or check the folder!")
