import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from collections import Counter
from datetime import datetime
import cv2
import time


# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Vehicle Detection",
    layout="centered"
)

st.title("🚗 Vehicle Detection (YOLOv11)")

# Check if running on Streamlit Cloud (check file path)
import os
current_file = os.path.abspath(__file__)
IS_CLOUD = "/mount/src/" in current_file.replace("\\", "/")  # Streamlit Cloud uses /mount/src/

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # Make sure best.pt is in same folder

model = load_model()

# Initialize session state
if 'captured_frame' not in st.session_state:
    st.session_state.captured_frame = None
if 'image_name' not in st.session_state:
    st.session_state.image_name = None
if 'camera_streaming' not in st.session_state:
    st.session_state.camera_streaming = False
if 'input_mode' not in st.session_state:
    st.session_state.input_mode = None
if 'photo_captured_flag' not in st.session_state:
    st.session_state.photo_captured_flag = False
if 'camera_obj' not in st.session_state:
    st.session_state.camera_obj = None
if 'camera_idx' not in st.session_state:
    st.session_state.camera_idx = None

# Function to find available camera
def get_available_camera():
    """Find first available camera device"""
    for i in range(10):  # Try cameras 0-9
        try:
            cap = cv2.VideoCapture(i)
            if cap is not None and cap.isOpened():
                # Try to read a frame to confirm it works
                ret, frame = cap.read()
                cap.release()
                if ret:
                    return i
        except Exception as e:
            continue
    return None

# Function to safely open camera
def open_camera(camera_idx):
    try:
        # Release old camera if exists
        if st.session_state.camera_obj is not None:
            st.session_state.camera_obj.release()
        
        cap = cv2.VideoCapture(camera_idx)
        if not cap.isOpened():
            return None
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        return cap
    except Exception as e:
        st.error(f"Failed to open camera: {e}")
        return None

# Function to safely close camera
def close_camera():
    try:
        if st.session_state.camera_obj is not None:
            st.session_state.camera_obj.release()
            st.session_state.camera_obj = None
    except:
        pass

# Function to capture photo from camera
def capture_photo():
    camera_idx = get_available_camera()
    
    if camera_idx is None:
        return False, "❌ No camera found. Please check camera connection."
    
    try:
        cap = cv2.VideoCapture(camera_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        # Warm up camera - read a few frames
        for _ in range(5):
            ret, _ = cap.read()
            if not ret:
                cap.release()
                return False, "Camera warm-up failed"
        
        # Capture frame
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return True, frame_rgb
        else:
            return False, "Failed to capture frame"
    except Exception as e:
        return False, f"Camera error: {str(e)}"

# ----------- Input Mode Selection ----------------
st.subheader("🎯 Select Input Mode")

# Debug: Show environment info
with st.expander("📊 Debug Info"):
    st.write(f"**Running Environment:** {'☁️ Cloud' if IS_CLOUD else '💻 Local'}")
    st.write(f"**Camera Available:** {'❌ No (Cloud)' if IS_CLOUD else '✅ Yes (Local)'}")

if IS_CLOUD:
    # Cloud environment - no camera available
    mode_col1, mode_col2 = st.columns(2)
    with mode_col1:
        if st.button("📁 Browse File", key="browse_mode_btn", use_container_width=True):
            st.session_state.input_mode = "browse"
            st.session_state.camera_streaming = False
            st.rerun()
    
    with mode_col2:
        if st.button("🗑️ Clear All", key="clear_all_btn", use_container_width=True):
            st.session_state.captured_frame = None
            st.session_state.image_name = None
            st.session_state.camera_streaming = False
            st.session_state.input_mode = None
            st.session_state.photo_captured_flag = False
            st.rerun()
    
    if st.session_state.input_mode is None:
        st.info("ℹ️ **Camera not available on cloud.** Use 'Browse File' to upload images.")
else:
    # Local environment - camera is available
    mode_col1, mode_col2, mode_col3 = st.columns(3)
    
    with mode_col1:
        if st.button("📁 Browse File", key="browse_mode_btn", use_container_width=True):
            st.session_state.input_mode = "browse"
            st.session_state.camera_streaming = False
            st.rerun()
    
    with mode_col2:
        if st.button("📷 Take Photo", key="take_photo_mode_btn", use_container_width=True):
            st.session_state.input_mode = "camera"
            st.session_state.photo_captured_flag = False
            st.rerun()
    
    with mode_col3:
        if st.button("🗑️ Clear All", key="clear_all_btn", use_container_width=True):
            st.session_state.captured_frame = None
            st.session_state.image_name = None
            st.session_state.camera_streaming = False
            st.session_state.input_mode = None
            st.session_state.photo_captured_flag = False
            st.rerun()
    
    if st.session_state.input_mode is None:
        st.info("👆 **Select an input mode above:**\n- 📁 Browse File: Upload an image\n- 📷 Take Photo: Capture from camera")

# ----------- Browse File Mode ----------------
if st.session_state.input_mode == "browse":
    st.divider()
    st.subheader("📁 Choose Image File")
    
    uploaded_file = st.file_uploader(
        "Select an image file",
        type=["jpg", "jpeg", "png"],
        key="file_uploader"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        frame_rgb = np.array(image)
        
        # Generate image name from file
        image_name = uploaded_file.name
        st.session_state.image_name = image_name
        st.session_state.captured_frame = frame_rgb
        
        st.divider()
        st.subheader("🖼️ Image Loaded - Ready for Detection")
        st.image(frame_rgb, caption=f"📸 {image_name}", use_container_width=True)
        
        st.write(f"**Image Name:** `{image_name}`")
        st.write("**Status:** ✅ Ready for detection analysis")
        
        if st.button("🔍 Detection Image", key="detect_btn_browse", use_container_width=True):
            with st.spinner("🤖 Analyzing image..."):
                # Ensure image is a contiguous uint8 numpy array for torch.from_numpy
                img_in = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
                try:
                    results = model.predict(
                        source=img_in,
                        conf=0.25,
                        device="cpu"
                    )
                except Exception as e:
                    st.error(f"Model prediction error: {e}")
                    results = None

            if results is not None:
                st.success("✅ Detection analysis complete!")

                # Show detection result
                annotated = results[0].plot()
                st.image(annotated, caption="Detection Result", use_container_width=True)

                # Extract detection information
                boxes = results[0].boxes

                if boxes is not None and len(boxes) > 0:
                    
                    class_ids = boxes.cls.tolist()
                    class_names = [model.names[int(cls)] for cls in class_ids]
                    confidences = boxes.conf.tolist()
                    counts = Counter(class_names)
                    
                    st.subheader("📊 Detection Results")
                    st.write(f"🖼️ **Image Name:** `{image_name}`")
                    st.write(f"✅ **Objects Detected:** {sum(counts.values())}")
                    
                    # Show detailed detections with accuracy/confidence
                    st.write("**Detection Details:**")
                    for i, (class_name, confidence) in enumerate(zip(class_names, confidences), 1):
                        accuracy_percent = confidence * 100
                        st.write(f"{i}. **{class_name}** → Accuracy: {accuracy_percent:.2f}%")
                    
                    # Summary by class with average accuracy
                    st.write("**Summary:**")
                    for name in sorted(counts.keys()):
                        class_confidences = [confidences[j] for j, cn in enumerate(class_names) if cn == name]
                        avg_accuracy = (sum(class_confidences) / len(class_confidences)) * 100
                        st.write(f"• **{name}**: {counts[name]} detected → Avg Accuracy: {avg_accuracy:.2f}%")
                else:
                    st.warning("⚠️ No objects detected in the image")

# ----------- Camera Mode ----------------
elif st.session_state.input_mode == "camera":
    st.divider()
    st.subheader("📷 Capture Image from Camera")
    
    col_camera1, col_camera2 = st.columns(2)
    
    with col_camera1:
        if st.button("📷Camera", key="open_camera_btn", use_container_width=True):
            st.session_state.camera_streaming = True
            st.session_state.photo_captured_flag = False
            st.rerun()
    
    with col_camera2:
        if st.button("⏹️ Close Camera", key="close_camera_btn", use_container_width=True):
            st.session_state.camera_streaming = False
            st.rerun()
    
    # Live camera preview
    if st.session_state.camera_streaming:
        st.write("**📹 Camera is OPEN - Click 'Click Photo' when ready**")
        
        # Show buttons outside of loop for stability
        col_photo1, col_photo2 = st.columns(2)
        capture_btn = False
        stop_btn = False
        
        with col_photo1:
            if st.button("✅ Click Photo", key="capture_photo_btn", use_container_width=True):
                capture_btn = True
        
        with col_photo2:
            if st.button("⏹️ Stop", key="stop_btn", use_container_width=True):
                close_camera()
                st.session_state.camera_streaming = False
                st.rerun()
        
        # Initialize camera if not already open
        if st.session_state.camera_obj is None:
            camera_idx = get_available_camera()
            st.session_state.camera_idx = camera_idx
            
            if camera_idx is None:
                st.error("❌ No camera found!")
                st.session_state.camera_streaming = False
                st.rerun()
            else:
                cap = open_camera(camera_idx)
                if cap is None:
                    st.error("❌ Camera failed to open!")
                    st.session_state.camera_streaming = False
                    st.rerun()
                else:
                    st.session_state.camera_obj = cap
        
        # Camera streaming loop
        if st.session_state.camera_obj is not None:
            try:
                cap = st.session_state.camera_obj
                preview_placeholder = st.empty()
                
                frame_count = 0
                last_time = time.time()
                target_fps = 15
                frame_delay = 1.0 / target_fps
                
                while st.session_state.camera_streaming and frame_count < 500:
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        preview_placeholder.image(frame_rgb, caption="📹 Live Camera Feed", use_container_width=True)
                        
                        # Check if capture button was clicked
                        if capture_btn:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            st.session_state.image_name = f"captured_{timestamp}"
                            st.session_state.captured_frame = frame_rgb.copy()
                            close_camera()
                            st.session_state.camera_streaming = False
                            st.success("✅ Photo Captured! Ready for Detection")
                            st.rerun()
                        
                        # Control frame rate
                        elapsed = time.time() - last_time
                        sleep_time = frame_delay - elapsed
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        last_time = time.time()
                        frame_count += 1
                    else:
                        st.warning("⚠️ Camera lost connection")
                        close_camera()
                        st.session_state.camera_streaming = False
                        st.rerun()
                
                close_camera()
                
            except Exception as e:
                st.error(f"Camera error: {str(e)}")
                close_camera()
                st.session_state.camera_streaming = False
    
    # Display captured image if available
    if st.session_state.captured_frame is not None:
        
        image_name = st.session_state.image_name
        frame_rgb = st.session_state.captured_frame
        
        st.divider()
        st.subheader("🖼️ Photo Captured - Ready for Detection")
        st.image(frame_rgb, caption=f"📸 {image_name}", use_container_width=True)
        
        st.write(f"**Image Name:** `{image_name}`")
        st.write("**Status:** ✅ Ready for detection analysis")
        
        if st.button("🔍 Detection Image", key="detect_btn_camera", use_container_width=True):
            
            with st.spinner("🤖 Analyzing image..."):
                # Ensure image is contiguous uint8 numpy array
                img_in = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
                try:
                    results = model.predict(
                        source=img_in,
                        conf=0.25,
                        device="cpu"
                    )
                except Exception as e:
                    st.error(f"Model prediction error: {e}")
                    results = None
            
            if results is not None:
                st.success("✅ Detection analysis complete!")
                
                # Show detection result
                annotated = results[0].plot()
                st.image(annotated, caption="Detection Result", use_container_width=True)
                
                # Extract detection information
                boxes = results[0].boxes
                
                if boxes is not None and len(boxes) > 0:
                    
                    class_ids = boxes.cls.tolist()
                    class_names = [model.names[int(cls)] for cls in class_ids]
                    confidences = boxes.conf.tolist()
                    counts = Counter(class_names)
                    
                    st.subheader("📊 Detection Results")
                    st.write(f"🖼️ **Image Name:** `{image_name}`")
                    st.write(f"✅ **Objects Detected:** {sum(counts.values())}")
                    
                    # Show detailed detections with accuracy/confidence
                    st.write("**Detection Details:**")
                    for i, (class_name, confidence) in enumerate(zip(class_names, confidences), 1):
                        accuracy_percent = confidence * 100
                        st.write(f"{i}. **{class_name}** → Accuracy: {accuracy_percent:.2f}%")
                    
                    # Summary by class with average accuracy
                    st.write("**Summary:**")
                    for name in sorted(counts.keys()):
                        class_confidences = [confidences[j] for j, cn in enumerate(class_names) if cn == name]
                        avg_accuracy = (sum(class_confidences) / len(class_confidences)) * 100
                        st.write(f"• **{name}**: {counts[name]} detected → Avg Accuracy: {avg_accuracy:.2f}%")
                else:
                    st.warning("⚠️ No objects detected in the image")
