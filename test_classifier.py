import cv2
import mediapipe as mp
import numpy as np
import pickle

# Load the trained model
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Dictionary to map labels to characters
labels_dict ={
    'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G',
    'H': 'H', 'I': 'I', 'J': 'J', 'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N',
    'O': 'O', 'P': 'P', 'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T', 'U': 'U',
    'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y', 'Z': 'Z',
    'del': 'del', 'nothing': 'nothing', 'space': 'space'
}

# Use a context manager for better resource handling
with mp_hands.Hands(
    static_image_mode=False,  # Set to False for real-time video
    max_num_hands=1,          # Limit to 1 hand to avoid feature mismatch
    min_detection_confidence=0.5,  # Confidence threshold for detection
    min_tracking_confidence=0.5    # Confidence threshold for tracking
) as hands:

    # Open the webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        data_aux = []
        x_ = []
        y_ = []
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Get frame dimensions
        H, W, _ = frame.shape

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        results = hands.process(frame_rgb)

        # Draw hand landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Extract hand landmarks
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                    x_.append(x)
                    y_.append(y)

                # Ensure only one hand's landmarks are processed
                break  # Exit after processing the first hand

            # Calculate bounding box coordinates
            x1 = int(min(x_) * W)
            y1 = int(min(y_) * H)
            x2 = int(max(x_) * W)
            y2 = int(max(y_) * H)

            # Make prediction
            if len(data_aux) == 42:  # Ensure correct number of features
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[prediction[0]]

                # Draw bounding box and predicted character
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
            else:
                print("Incorrect number of features. Skipping prediction.")

        # Display the frame
        cv2.imshow('Hand Tracking', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()