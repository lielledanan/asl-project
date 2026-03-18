import cv2
import tensorflow as tf
import numpy as np
from collections import deque, Counter
import os

# ביטול הודעות מערכת של טנזורפלו
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ===== 1. טעינת המודל =====
model = tf.keras.models.load_model("asl_recognition_model.keras")

# ===== 2. רשימת האותיות המדויקת מהקולב =====
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
               'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# ===== 3. משתני עזר לזיהוי יציב =====
buffer = deque(maxlen=12)
final_text = ""
last_added_letter = None

# ===== 4. פתיחת מצלמה =====
cap = cv2.VideoCapture(0)

print("המערכת פועלת! לחצי על 'q' ליציאה ועל 'r' לאיפוס הטקסט.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # הגדרת אזור העבודה (ROI)
    x1, y1, x2, y2 = 100, 100, 350, 350
    roi = frame[y1:y2, x1:x2]

    # --- עיבוד התמונה (Preprocessing) ---
    img = cv2.resize(roi, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # המרה ל-float32 (אם זה עדיין מתבלבל, אפשר לנסות להוסיף חלוקה ב-255 בסוף השורה)
    img_input = img.astype("float32")
    img_input = np.expand_dims(img_input, axis=0)

    # --- חיזוי המודל ---
    prediction = model.predict(img_input, verbose=0)
    class_id = np.argmax(prediction)
    confidence = prediction[0][class_id]

    # שיפור: קביעת האות לפי סף ביטחון
    if confidence < 0.3:
        current_letter = "nothing"
    else:
        current_letter = class_names[class_id]

    # --- לוגיקה להוספת אותיות למחרוזת (בלי חזרות מיותרות) ---
    if current_letter != "nothing" and confidence > 0.70:
        buffer.append(current_letter)

        if len(buffer) == 12:
            most_common, count = Counter(buffer).most_common(1)[0]

            # הוספת האות רק אם היא הופיעה ברוב הפריימים והיא לא האחרונה שהוספנו
            if count >= 10 and most_common != last_added_letter:
                if most_common == "space":
                    final_text += " "
                elif most_common == "del":
                    final_text = final_text[:-1]
                else:
                    final_text += most_common

                last_added_letter = most_common
                buffer.clear()  # ניקוי כדי למנוע כפילויות רצופות

    # איפוס הזיכרון אם המודל מזהה 'nothing' לאורך זמן
    if current_letter == "nothing":
        last_added_letter = None

    # --- תצוגה גרפית על המסך ---
    # ציור המסגרת
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # הצגת האות הנוכחית והביטחון
    label = f"{current_letter} ({confidence:.2f})"
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # הצגת הטקסט שנכתב בתחתית המסך
    cv2.rectangle(frame, (0, 400), (frame.shape[1], 480), (255, 255, 255), -1)  # רקע לבן לטקסט
    cv2.putText(frame, f"Result: {final_text}", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    cv2.imshow("ASL Real-Time Translator", frame)

    # מקשי שליטה
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        final_text = ""
        last_added_letter = None

cap.release()
cv2.destroyAllWindows()