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
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def toggle_processing(self):
        self.apply_processing = not self.apply_processing

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Resize the frame to increase the display size
        img = cv2.resize(img, (800, 600))  # Resize to 800x600

        if self.apply_processing:
            # Detect faces in the frame
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # Blur each face found in the frame
            for (x, y, w, h) in faces:
                face = img[y:y+h, x:x+w]
                face = cv2.GaussianBlur(face, (99, 99), 30)
                img[y:y+h, x:x+w] = face
        
        return img

# Streamlit UI
st.title("Video Streaming with WebRTC - Face Blur")

# Option to toggle video processing
processing_toggle = st.checkbox("Apply Face Blur Processing")

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
