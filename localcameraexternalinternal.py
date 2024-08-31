import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np

# Optional: Set up WebRTC configuration for STUN/TURN servers if needed
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Custom VideoProcessor class to process video frames
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.function = "None"
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def set_function(self, function):
        self.function = function

    def apply_face_blur(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            face = cv2.GaussianBlur(face, (99, 99), 30)
            img[y:y+h, x:x+w] = face
        return img

    def apply_edge_detection(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return edges_colored

    def apply_sepia_filter(self, img):
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        sepia_img = cv2.transform(img, sepia_filter)
        sepia_img = np.clip(sepia_img, 0, 255)
        return sepia_img

    def flip_video(self, img):
        return cv2.flip(img, 1)  # Flip horizontally

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (800, 600))  # Resize to 800x600

        if self.function == "Face Blur":
            img = self.apply_face_blur(img)
        elif self.function == "Edge Detection":
            img = self.apply_edge_detection(img)
        elif self.function == "Sepia Filter":
            img = self.apply_sepia_filter(img)
        elif self.function == "Flip Video":
            img = self.flip_video(img)

        return img

# Streamlit UI
st.title("Video Streaming with WebRTC - Camera Functions")

# Dropdown to select the camera function
function = st.selectbox("Select Camera Function", ["None", "Face Blur", "Edge Detection", "Sepia Filter", "Flip Video"])

# Initialize the WebRTC streamer
webrtc_ctx = webrtc_streamer(
    key="camera-functions",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True
)

# Apply the selected function
if webrtc_ctx.video_processor:
    webrtc_ctx.video_processor.set_function(function)
