import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2

# Optional: Set up WebRTC configuration for STUN/TURN servers if needed
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Custom VideoProcessor class to process video frames
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.apply_processing = False

    def toggle_processing(self):
        self.apply_processing = not self.apply_processing

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if self.apply_processing:
            # Example: Convert to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert back to BGR for display
        
        return img

# Streamlit UI
st.title("Video Streaming with WebRTC")

# Option to toggle video processing
processing_toggle = st.checkbox("Apply Grayscale Processing")

# Initialize the WebRTC streamer
webrtc_ctx = webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True
)

# Apply processing if the checkbox is selected
if webrtc_ctx.video_processor:
    if processing_toggle:
        webrtc_ctx.video_processor.toggle_processing()

# Note: WebRTC works on both desktop and mobile browsers.
