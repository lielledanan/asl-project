import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
import tensorflow as tf
import numpy as np
from collections import deque, Counter

# 1. Page Configuration
st.set_page_config(page_title="ASL Real-Time Translator", layout="wide")

st.markdown("""
   <style>
   .main { background-color: #f5f7f9; }
   .ststInfo { background-color: #ffffff; border-radius: 10px; padding: 20px; border: 1px solid #e0e0e0; }
   h1 { color: #2c3e50; font-family: 'Helvetica Neue', sans-serif; }
   </style>
   """, unsafe_allow_html=True)

# 2. Sidebar & Model Loading
st.sidebar.title("Configuration & Guide")
st.sidebar.info("This app uses a CNN model to translate ASL fingerspelling into English text in real-time.")

try:
    st.sidebar.image("guide.png", caption="ASL Fingerspelling Guide", use_container_width=True)
except:
    st.sidebar.warning("Note: Place 'guide.png' in your project folder.")


@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("asl_recognition_model.keras")


model = load_my_model()
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
               'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']


# 3. Balanced Video Processor
class ASLVideoProcessor:
    def __init__(self):
        self.buffer = deque(maxlen=10)
        self.last_added_letter = None
        self.frame_count = 0
        self.final_text = ""
        self.display_label = "Scanning..."

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        self.frame_count += 1

        x1, y1, x2, y2 = 100, 100, 350, 350
        roi = img[y1:y2, x1:x2]

        # דילוג מינימלי על פריימים (3) כדי לשמור על מהירות וגם על איכות הסיווג
        if self.frame_count % 3 == 0:
            processed = cv2.resize(roi, (64, 64))
            rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(rgb.astype("float32"), axis=0)

            prediction = model.predict(input_data, verbose=0)
            class_id = np.argmax(prediction)
            confidence = prediction[0][class_id]

            # ספי הביטחון שביקשת
            letter = class_names[class_id] if confidence > 0.30 else "nothing"

            if letter != "nothing" and confidence > 0.70:
                self.buffer.append(letter)
                if len(self.buffer) == 10:
                    most_common, count = Counter(self.buffer).most_common(1)[0]
                    if count >= 8 and most_common != self.last_added_letter:
                        if most_common == "space":
                            self.final_text += " "
                        elif most_common == "del":
                            self.final_text = self.final_text[:-1]
                        else:
                            self.final_text += most_common
                        self.last_added_letter = most_common
                        self.buffer.clear()

            if letter == "nothing":
                self.last_added_letter = None

            self.display_label = f"{letter} ({confidence:.2f})"

        # ציור הממשק בתוך הווידאו
        cv2.rectangle(img, (x1, y1), (x2, y2), (46, 204, 113), 2)
        cv2.putText(img, self.display_label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (46, 204, 113), 2)

        overlay = img.copy()
        cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        cv2.putText(img, f"Result: {self.final_text}", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return frame.from_ndarray(img, format="bgr24")


# 4. Main UI Layout
st.title("🤟 ASL Real-Time Translator")
st.write("Hold your hand signs within the green box to start translating")

webrtc_streamer(
    key="asl-translator",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    video_processor_factory=ASLVideoProcessor,
    async_processing=True,
)

st.markdown("---")
st.caption("Developed with ❤️ for Machine Learning Project 2026- Lielle Danan")