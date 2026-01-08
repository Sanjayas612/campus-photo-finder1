import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import cv2
import mediapipe as mp
import numpy as np
import face_recognition
import zipfile
from io import BytesIO

# Initialize MediaPipe for Head Pose Tracking
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

st.set_page_config(page_title="Pro Campus Photo Finder", layout="wide")
st.title("🛡️ Universal Pro Face Scanner")

# Session State to hold learned "Face DNA" from multiple angles
if 'face_dna_list' not in st.session_state:
    st.session_state.face_dna_list = []

# --- 1. FOLDER UPLOAD SECTION ---
st.header("1️⃣ Select Event Folder")
st.write("Upload the images you want to search through.")
uploaded_files = st.file_uploader("Upload Event Photos", accept_multiple_files=True, type=['jpg', 'png', 'jpeg'])

# --- 2. LIVE SCANNER SECTION ---
st.header("2️⃣ Multi-Angle Face Scan")
st.info("Rotate your head slowly: Center -> Left -> Right. The AI will learn your profile.")

# Tracking angles captured
captured_angles = st.empty()

class FaceScanner:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)
        
        status_msg = "Align your face..."
        color = (0, 0, 255) # Red

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            # Nose tip (1), Left eye (33), Right eye (263)
            nose_x = landmarks[1].x
            
            # Identify Angle
            if nose_x < 0.4:
                angle = "Right Profile"
            elif nose_x > 0.6:
                angle = "Left Profile"
            else:
                angle = "Center"
            
            status_msg = f"Scanning: {angle} - STAY STILL"
            color = (0, 255, 0) # Green

            # Automatically 'Learn' the face if we haven't already
            # We take a sample every few frames to build the DNA list
            if len(st.session_state.face_dna_list) < 15: # Cap it at 15 samples
                encodings = face_recognition.face_encodings(img_rgb)
                if encodings:
                    st.session_state.face_dna_list.append(encodings[0])

        cv2.putText(img, status_msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return img

# Launch WebRTC Streamer
webrtc_streamer(key="scanner", video_processor_factory=FaceScanner)

# --- 3. MATCHING ENGINE ---
if st.button("🚀 Start Search with Learned Profile"):
    if not st.session_state.face_dna_list:
        st.error("Please scan your face first!")
    elif not uploaded_files:
        st.error("Please upload event photos!")
    else:
        st.success(f"Profile Loaded! Learned {len(st.session_state.face_dna_list)} facial points.")
        
        matched_files = []
        progress_bar = st.progress(0)
        
        for i, file in enumerate(uploaded_files):
            # Load event image
            img = face_recognition.load_image_file(file)
            event_encodings = face_recognition.face_encodings(img)
            
            for e_enc in event_encodings:
                # Compare against ALL captured angles in the DNA list
                # This makes it very flexible for side shots!
                matches = face_recognition.compare_faces(st.session_state.face_dna_list, e_enc, tolerance=0.53)
                if True in matches:
                    matched_files.append(file)
                    break
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        if matched_files:
            st.balloons()
            st.success(f"✨ Found {len(matched_files)} photos of you!")
            
            # Preview and Download
            cols = st.columns(4)
            for idx, m in enumerate(matched_files):
                cols[idx % 4].image(m, use_container_width=True)

            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                for f in matched_files:
                    zf.writestr(f.name, f.getvalue())
            
            st.download_button("📥 Download All My Photos (.zip)", data=zip_buffer.getvalue(), file_name="campus_matches.zip")
        else:
            st.warning("No matches found. Try a more thorough face scan!")
