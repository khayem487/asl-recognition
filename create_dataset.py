import os
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import pickle

# Initialize MediaPipe components
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Use a context manager for better resource handling
with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        model_complexity=1,  # Added in newer versions (0=light, 1=full, 2=heavy)
        min_detection_confidence=0.3
) as hands:
    data_dir = './data2'

    data=[]
    labels=[]
    for dir_name in os.listdir(data_dir):
        dir_path = os.path.join(data_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue
        print(dir_path)
        # Process first image in directory
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Skipping invalid image: {img_path}")
                continue

            data_aux=[]
            # Convert to RGB and ensure contiguous memory
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb = np.ascontiguousarray(img_rgb)

            # Process image
            results = hands.process(img_rgb)

            # Draw on a BGR copy
            annotated_image = img.copy()  # BGR format for OpenCV
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x=hand_landmarks.landmark[i].x
                        y=hand_landmarks.landmark[i].y
                        data_aux.append(x)
                        data_aux.append(y)

                data.append(data_aux)
                labels.append(dir_name)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels},f)
f.close()