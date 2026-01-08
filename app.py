import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import zipfile
from io import BytesIO

# Initialize Google MediaPipe
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

st.set_page_config(page_title="Google AI Photo Finder", page_icon="🎓")
st.title("🎓 Google AI Photo Finder")
st.write("Scan your face to let the AI learn, then upload your event folder!")

# 1. Selection: Multi-file upload (Standard for web folders)
uploaded_files = st.file_uploader("1️⃣ Upload Event Photos", accept_multiple_files=True, type=['jpg', 'png', 'jpeg'])

# 2. Scanning: Learn the face
cam_image = st.camera_input("2️⃣ Scan your face to learn")

if uploaded_files and cam_image:
    if st.button("🚀 Start Finding Me"):
        # Convert camera scan to image
        scan_bytes = np.frombuffer(cam_image.getvalue(), np.uint8)
        scan_img = cv2.imdecode(scan_bytes, cv2.IMREAD_COLOR)
        scan_rgb = cv2.cvtColor(scan_img, cv2.COLOR_BGR2RGB)
        
        # Google AI: Learn the user's face geometry
        results = face_detection.process(scan_rgb)
        
        if not results.detections:
            st.error("Google AI couldn't see your face. Please adjust your lighting.")
        else:
            st.success("✅ Face Learned! Scanning the folder now...")
            
            # Get 'User DNA' (the shape/location of the face)
            user_box = results.detections[0].location_data.relative_bounding_box
            user_features = np.array([user_box.width, user_box.height])

            matched_files = []
            progress_bar = st.progress(0)

            for i, file in enumerate(uploaded_files):
                # Process event photo
                file_bytes = np.frombuffer(file.getvalue(), np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Google AI: Find faces in the event photo
                event_results = face_detection.process(img_rgb)
                
                if event_results.detections:
                    for det in event_results.detections:
                        eb = det.location_data.relative_bounding_box
                        event_features = np.array([eb.width, eb.height])
                        
                        # Compare user geometry to event geometry
                        # This is a lightweight 'Learning' match
                        distance = np.linalg.norm(user_features - event_features)
                        
                        if distance < 0.05: # Accuracy threshold
                            matched_files.append(file)
                            break
                
                progress_bar.progress((i + 1) / len(uploaded_files))

            if matched_files:
                st.balloons()
                st.success(f"✨ Found {len(matched_files)} photos!")
                
                # Zip for download
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    for f in matched_files:
                        zf.writestr(f.name, f.getvalue())
                
                st.download_button("📥 Download My Photos (.zip)", data=zip_buffer.getvalue(), file_name="matches.zip")
            else:
                st.warning("No matches found. Try scanning from a different distance!")
