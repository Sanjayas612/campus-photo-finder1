import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import numpy as np
import zipfile
from io import BytesIO

st.set_page_config(page_title="Campus Photo Finder", layout="centered")
st.title("🎓 Smart Campus Photo Finder")

# Session State to store the 'Learned' facial data
if 'user_face_samples' not in st.session_state:
    st.session_state.user_face_samples = []

# 1. Folder Upload
uploaded_files = st.file_uploader("1️⃣ Upload Event Photos", accept_multiple_files=True, type=['jpg','jpeg','png'])

# 2. Live Scanner
st.header("2️⃣ Live Face Scan")
st.info("The AI is waiting. Move your head slowly when the camera starts.")

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Using a very light face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            # Draw box and instruction
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, "SCANNING...", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save a sample of the face 'DNA' (Histogram)
            roi_gray = gray[y:y+h, x:x+w]
            hist = cv2.calcHist([roi_gray], [0], None, [256], [0, 256])
            if len(st.session_state.user_face_samples) < 20:
                st.session_state.user_face_samples.append(hist)

        return img

webrtc_streamer(key="scanner", video_processor_factory=VideoProcessor)

# 3. Matching
if st.button("🚀 Start Matching My Face"):
    if not st.session_state.user_face_samples:
        st.error("Scan your face first!")
    elif not uploaded_files:
        st.error("Upload photos first!")
    else:
        matched_files = []
        progress = st.progress(0)
        
        for i, file in enumerate(uploaded_files):
            file_bytes = np.frombuffer(file.getvalue(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            
            # Simple but fast histogram comparison
            img_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            
            for sample_hist in st.session_state.user_face_samples:
                score = cv2.compareHist(sample_hist, img_hist, cv2.HISTCMP_CORREL)
                if score > 0.85: # High matching threshold
                    matched_files.append(file)
                    break
            progress.progress((i + 1) / len(uploaded_files))
            
        if matched_files:
            st.success(f"✨ Found {len(matched_files)} matches!")
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                for f in matched_files:
                    zf.writestr(f.name, f.getvalue())
            st.download_button("📥 Download Zip", zip_buffer.getvalue(), "matches.zip")
        else:
            st.warning("No matches found. Try scanning from a different angle.")
