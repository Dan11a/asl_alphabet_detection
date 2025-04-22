import cv2
import numpy as np
import mediapipe as mp
import joblib
import time

clf = joblib.load("RFC_model3.pkl")
le = joblib.load("label_encoder.pkl")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
prev_time = time.time()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    prediction = "No Hand"

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

        # BBX for x,y
        x_vals = [pt[0] for pt in landmarks]
        y_vals = [pt[1] for pt in landmarks]

        min_x, max_x = min(x_vals), max(x_vals)
        min_y, max_y = min(y_vals), max(y_vals)

        normalized = [
            (
                (x - min_x) / (max_x - min_x + 1e-6),
                (y - min_y) / (max_y - min_y + 1e-6),
                z 
            )
            for (x, y, z) in landmarks
        ]

        flattened = np.array([val for pt in normalized for val in pt]).reshape(1, -1)

        if flattened.shape[1] == 63:        # only predict if all landmarks are detected
            pred_idx = clf.predict(flattened)[0]
            prediction = le.inverse_transform([pred_idx])[0]

        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if time.time() - prev_time >= 3.0:
            print("Predicted Letter:", prediction)  
            prev_time = time.time()

    cv2.putText(frame, f'Prediction: {prediction}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



    cv2.imshow("ASL Letter Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()

