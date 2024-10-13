import cv2
import mediapipe as mp
import numpy as np
from keras.models import model_from_json
from keras.models import Sequential, model_from_json
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import cv2
import numpy as np
import json
import keras
import sys

# Load the model JSON and weights
with open('sign_language_detection_model_l.json', 'r') as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)
model.load_weights("sign_language_detection_model_l.h5")

# Compile the model (if necessary)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Open the webcam
cap = cv2.VideoCapture(0)

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'blank']

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    result = hands.process(rgb_frame)

    # Check if hand is detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the bounding box for the detected hand
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])

            # Extract the hand region
            hand_roi = frame[y_min:y_max, x_min:x_max]

            # Preprocess the hand region
            hand_roi_gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
            hand_roi_resized = cv2.resize(hand_roi_gray, (48, 48))
            hand_roi_normalized = hand_roi_resized / 255.0
            hand_roi_reshaped = hand_roi_normalized.reshape(1, 48, 48, 1)

            # Predict the gesture
            pred = model.predict(hand_roi_reshaped)
            prediction_label = labels[pred.argmax()]
            accuracy = "{:.2f}".format(np.max(pred) * 100)

            # Display the predicted label and accuracy
            cv2.rectangle(frame, (x_min, y_min-30), (x_max, y_min), (0, 165, 255), -1)
            display_text = f"{prediction_label}  {accuracy}%" if prediction_label != 'blank' else " "
            cv2.putText(frame, display_text, (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Sign Language Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
hands.close()
