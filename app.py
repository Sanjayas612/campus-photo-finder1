import streamlit as st
import face_recognition
import cv2
import numpy as np
import zipfile              # Added missing import
from io import BytesIO      # Added missing import
from PIL import Image, ImageOps

# 1. Page config MUST be the very first Streamlit command and only once
st.set_page_config(page_title="Campus Photo Finder", layout="wide")

# 2. Helper function to fix orientation (Defined only ONCE)
def fix_image_orientation(img_file):
    """Checks for hidden EXIF rotation tags and fixes the image orientation."""
    image = Image.open(img_file)
    fixed_image = ImageOps.exif_transpose(image)
    return fixed_image.convert("RGB")

# 3. Safe CSS loading
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception as e:
    # This will show an error on screen if style.css is missing
    st.info("Note: Running without custom style.css")

# 4. UI Headers
st.title("🎓 Campus Photo Finder")
st.markdown("### Find your photos in seconds using AI")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("⚙️ Settings")
tolerance = st.sidebar.slider(
    "AI Sensitivity (Tolerance)", 
    0.4, 0.7, 0.58, 
    help="Lower = Stricter (fewer wrong matches). Higher = Looser (finds more photos)."
)

st.sidebar.info("Tip: If you get 'wrong' people, move the slider to 0.45.")

# --- STEP 1: UPLOAD REFERENCE ---
st.header("1️⃣ Upload Your Face")
ref_files = st.file_uploader("Upload 3+ clear photos of yourself", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

# --- STEP 2: UPLOAD EVENT FOLDER ---
st.header("2️⃣ Upload Event Photos")
event_files = st.file_uploader("Drag and drop all photos from the event here", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

# --- PROCESSING LOGIC ---
if ref_files and event_files:
    if st.button("🚀 Start Deep Search"):
        my_dna = []
        with st.spinner("Analyzing your face DNA..."):
            for ref in ref_files:
                fixed_ref = fix_image_orientation(ref)
                img = np.array(fixed_ref)
                encs = face_recognition.face_encodings(img)
                if encs:
                    my_dna.append(encs[0])
        
        if not my_dna:
            st.error("❌ Could not detect a face in your reference photos!")
        else:
            found_photos = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, event_photo in enumerate(event_files):
                status_text.text(f"Deep Scanning photo {i+1} of {len(event_files)}...")
                
                fixed_event = fix_image_orientation(event_photo)
                img = np.array(fixed_event)
                
                found_in_this_photo = False
                for angle in [0, 90]:
                    if found_in_this_photo: break
                    
                    if angle == 90:
                        processed_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    else:
                        processed_img = img

                    # Speed Hack: Shrink for scanning
                    small_img = cv2.resize(processed_img, (0, 0), fx=0.25, fy=0.25)
                    
                    # Detect and encode
                    face_locations_small = face_recognition.face_locations(small_img, model="hog")
                    
                    if face_locations_small:
                        scale = 4 # Scale back up from 0.25
                        face_locations = [
                            (top * scale, right * scale, bottom * scale, left * scale)
                            for (top, right, bottom, left) in face_locations_small
                        ]
                        face_encs = face_recognition.face_encodings(processed_img, face_locations)
                    
                        for enc in face_encs:
                            matches = face_recognition.compare_faces(my_dna, enc, tolerance=tolerance)
                            if sum(matches) >= (len(my_dna) / 2):
                                found_photos.append(event_photo)
                                found_in_this_photo = True
                                break
                
                progress_bar.progress((i + 1) / len(event_files))
            
            status_text.empty()
            
            if found_photos:
                st.balloons()
                st.success(f"✨ Found you in {len(found_photos)} photos!")
                
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    for p in found_photos:
                        p.seek(0)
                        zf.writestr(p.name, p.getvalue())
                
                st.download_button(
                    label="📥 Download My Matches (.zip)",
                    data=zip_buffer.getvalue(),
                    file_name="my_campus_photos.zip",
                    mime="application/zip"
                )
                
                st.markdown("---")
                cols = st.columns(4)
                for idx, p in enumerate(found_photos):
                    cols[idx % 4].image(p, use_container_width=True)
            else:
                st.warning("🔎 No matches found. Try moving the Sensitivity slider.")
