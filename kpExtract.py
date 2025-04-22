import mediapipe as mp
import cv2
import os
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

DATA_DIR = 'asl_alphabet_train'
LM_DIR = 'landmarks_dataset'

os.makedirs(LM_DIR, exist_ok=True)

for label in os.listdir(DATA_DIR):
    label_dir = os.path.join(DATA_DIR, label)
    lm_label_path = os.path.join(LM_DIR, label)
    os.makedirs(lm_label_path, exist_ok=True)
    
    for img_name in os.listdir(label_dir):
        img_path = os.path.join(label_dir, img_name)
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            landmarks = []
            for lm in results.multi_hand_landmarks[0].landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            np.save(os.path.join(lm_label_path, img_name.replace('.jpg', '.npy')), np.array(landmarks))
