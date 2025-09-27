import os
import av
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import time
#streamlit run realTime.py

# --- Page Setup ---
st.set_page_config(page_title="YOLOv12 Real-Time Detection", layout="wide")
st.title("📷 YOLOv12 Real-Time Webcam Detection")
st.markdown("Real-time object detection using **YOLOv12** inside WSL — no OpenCV window required.")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Settings")

# Model selection from models/ folder
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)  # ensure folder exists
available_models = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
if not available_models:
    st.sidebar.warning(f"No .pt files found in `{model_dir}/` folder.")
    st.stop()

model_choice = st.sidebar.selectbox("Select Classification Model", available_models, index=0)
conf_threshold = st.sidebar.slider("Detector Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
img_size = st.sidebar.selectbox("Image Size", [320, 480, 640, 800, 1080], index=2)

# Load YOLO model
model_path = os.path.join(model_dir, model_choice)
model = YOLO(model_path)

# --- Video Transformer ---
class YOLOv12Video(VideoTransformerBase):
    def __init__(self):
        self.prev_time = time.time()
    
    def transform(self, frame):
        
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time)
        self.prev_time = curr_time
        
        # Display FPS on the frame
        cv2.putText(frame.to_ndarray(format="bgr24"), f"FPS: {fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
        # Run YOLOv12 detection
        
        img = frame.to_ndarray(format="bgr24")
        results = model.predict(img, imgsz=img_size, conf=conf_threshold)
        annotated = results[0].plot()
        return annotated

# --- Main Webcam Display ---
st.markdown("### Live Webcam Feed")
st.info("If prompted, allow your browser to access the camera.")

webrtc_streamer(
    key="yolov12-stream",
    video_transformer_factory=YOLOv12Video,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True
)

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** Put your trained `.pt` files in the `models/` folder to use them in the dropdown.")
