import streamlit as st
import face_recognition
import os
import zipfile
from io import BytesIO
from PIL import Image

# Page Configuration
st.set_page_config(page_title="Universal Photo Finder", page_icon="🎓", layout="centered")

st.title("🎓 Campus Photo Finder")
st.write("Upload event photos and scan your face to find your memories instantly!")

# --- STEP 1: UPLOAD PHOTOS ---
st.header("1️⃣ Upload Event Photos")
uploaded_files = st.file_uploader(
    "Select the folder or multiple photos from the event", 
    accept_multiple_files=True, 
    type=['jpg', 'jpeg', 'png']
)

# --- STEP 2: LIVE FACE SCAN ---
st.header("2️⃣ Scan Your Face")
st.info("The AI needs to see your face to know who to look for.")
cam_image = st.camera_input("Take a photo of yourself")

# --- STEP 3: MATCHING LOGIC ---
if uploaded_files and cam_image:
    if st.button("🚀 Start Matching My Face"):
        # 1. Encode the user's face from the webcam
        user_image = face_recognition.load_image_file(cam_image)
        user_encodings = face_recognition.face_encodings(user_image)

        if not user_encodings:
            st.error("Could not detect your face. Please ensure good lighting and try again.")
        else:
            user_encoding = user_encodings[0]
            matched_images = []
            
            # Progress bar for the user
            progress_bar = st.progress(0)
            status_text = st.empty()

            # 2. Loop through all uploaded photos
            for i, file in enumerate(uploaded_files):
                img = face_recognition.load_image_file(file)
                # Find all faces in the event photo
                face_encs = face_recognition.face_encodings(img)
                
                for enc in face_encs:
                    # Tolerance 0.53 is the sweet spot we found earlier
                    match = face_recognition.compare_faces([user_encoding], enc, tolerance=0.53)
                    if match[0]:
                        matched_images.append(file)
                        break
                
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
                status_text.text(f"Scanning photo {i+1} of {len(uploaded_files)}...")

            # --- STEP 4: RESULTS & DOWNLOAD ---
            if matched_images:
                st.success(f"✨ Found {len(matched_images)} photos of you!")
                
                # Show a small preview grid
                st.subheader("Preview of your photos:")
                cols = st.columns(3)
                for idx, img_file in enumerate(matched_images):
                    cols[idx % 3].image(img_file, use_container_width=True)

                # Create ZIP file in memory
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    for img_file in matched_images:
                        zf.writestr(img_file.name, img_file.getvalue())
                
                st.download_button(
                    label="📥 Download My Photos (.zip)",
                    data=zip_buffer.getvalue(),
                    file_name="my_event_matches.zip",
                    mime="application/zip"
                )
            else:
                st.warning("No matches found. Try scanning your face again from a different angle!")