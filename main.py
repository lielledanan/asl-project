print("hi")
print("hello")

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque, Counter

# טען את המודל
model = load_model("asl_recognition_model.keras")

class_names = [
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z','del','nothing','space'
]

# buffer לזיהוי יציב יותר
buffer = deque(maxlen=8)
prev_letter = ""
final_text = ""

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # חיתוך ROI
    roi = frame[100:300, 100:300]
    resized = cv2.resize(roi, (64, 64))
    normalized = resized / 255.0
    input_img = np.expand_dims(normalized, axis=0)

    # חיזוי אות
    prediction = model.predict(input_img, verbose=0)
    confidence = np.max(prediction)
    class_id = np.argmax(prediction)
    letter = class_names[class_id]

    if confidence > 0.90:
        buffer.append(letter)
        if len(buffer) == 8:
            most_common = Counter(buffer).most_common(1)[0][0]
            if most_common != prev_letter:
                if most_common == "space":
                    final_text += " "
                elif most_common == "del":
                    final_text = final_text[:-1]
                elif most_common != "nothing":
                    final_text += most_common
                prev_letter = most_common
            buffer.clear()

    # ציור ROI וטקסט
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
    cv2.putText(frame, f"Letter: {letter}", (100, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Text: {final_text}", (50, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("ASL Translator", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("r"):
        final_text = ""  # reset text

cap.release()
cv2.destroyAllWindows()
