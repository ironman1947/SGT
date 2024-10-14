# check.py

import sys
import io
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
import logging
import os

# ----------------------------- #
#       Configure Encoding
# ----------------------------- #

# Set standard output to UTF-8 to handle Unicode characters
try:
    # For Python 3.7 and above
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    # For Python versions below 3.7
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ----------------------------- #
#       Configure Logging
# ----------------------------- #

# Configure logging to write to a file with UTF-8 encoding
logging.basicConfig(
    filename='gesture_detection.log',
    filemode='w',  # Overwrite the log file each run
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ----------------------------- #
#       Load Pretrained Model
# ----------------------------- #

# Paths to the saved model and artifacts
MODEL_PATH = 'hand_gesture_model.h5'
LABEL_ENCODER_PATH = 'label_encoder.pkl'
SCALER_PATH = 'scaler.pkl'

# Load the trained model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logging.info(f"Loaded model from '{MODEL_PATH}' successfully.")
except Exception as e:
    logging.error(f"Failed to load model from '{MODEL_PATH}': {e}")
    sys.exit(1)

# Load the Label Encoder and Scaler
try:
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    logging.info(f"Loaded label encoder from '{LABEL_ENCODER_PATH}' and scaler from '{SCALER_PATH}' successfully.")
except Exception as e:
    logging.error(f"Failed to load label encoder or scaler: {e}")
    sys.exit(1)

# ----------------------------- #
#       Initialize MediaPipe
# ----------------------------- #

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe Hands
hands = mp_hands.Hands(
    static_image_mode=False,       # For real-time detection
    max_num_hands=1,               # Detect one hand
    min_detection_confidence=0.5,  # Minimum confidence for detection
    min_tracking_confidence=0.5    # Minimum confidence for tracking
)

# ----------------------------- #
#       Initialize Webcam
# ----------------------------- #

cap = cv2.VideoCapture(0)  # 0 is the default webcam

if not cap.isOpened():
    logging.error("Error: Could not open webcam.")
    print("Error: Could not open webcam.")
    sys.exit(1)

# ----------------------------- #
#       Prediction Function
# ----------------------------- #

def predict_gesture(keypoints):
    """
    Preprocesses keypoints and predicts the gesture label.

    Args:
        keypoints (list or np.array): Flattened list of keypoint coordinates.

    Returns:
        str: Predicted gesture label.
    """
    try:
        # Convert to numpy array and reshape for scaler
        keypoints = np.array(keypoints).reshape(1, -1)

        # Scale the keypoints using the loaded scaler
        keypoints_scaled = scaler.transform(keypoints)

        # Predict using the loaded model
        predictions = model.predict(keypoints_scaled)
        predicted_class = np.argmax(predictions, axis=1)

        # Decode the predicted class to label
        predicted_label = label_encoder.inverse_transform(predicted_class)[0]

        return predicted_label

    except Exception as e:
        logging.error(f"Error in predict_gesture: {e}")
        return "Prediction_Error"

# ----------------------------- #
#       Main Loop
# ----------------------------- #

logging.info("=== Real-Time Hand Gesture Recognition Started ===")
print("=== Real-Time Hand Gesture Recognition ===")
print("Press 'ESC' to quit.\n")
logging.info("Press 'ESC' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        logging.warning("Failed to grab frame.")
        continue  # Skip to the next iteration

    # Flip the frame horizontally for a mirror view
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    result = hands.process(image_rgb)

    gesture = ''

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # Extract keypoints
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])

            # Predict the gesture
            gesture = predict_gesture(keypoints)

            # Log the prediction
            logging.info(f"Predicted Gesture: {gesture}")

            # Display the prediction on the frame
            try:
                cv2.putText(
                    frame,
                    f"Gesture: {gesture}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    3,
                    cv2.LINE_AA
                )
            except UnicodeEncodeError as e:
                logging.error(f"UnicodeEncodeError during putText: {e}")
                cv2.putText(
                    frame,
                    "Gesture: [Error]",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA
                )

    else:
        # If no hand is detected, display a message
        try:
            cv2.putText(
                frame,
                "No Hand Detected",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
                cv2.LINE_AA
            )
            logging.info("No Hand Detected")
        except UnicodeEncodeError as e:
            logging.error(f"UnicodeEncodeError during putText: {e}")

    # Display instructions on the frame
    try:
        cv2.putText(
            frame,
            "Press 'ESC' to quit.",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
    except UnicodeEncodeError as e:
        logging.error(f"UnicodeEncodeError during putText: {e}")

    # Show the frame
    cv2.imshow('Real-Time Hand Gesture Recognition', frame)

    # Exit if ESC is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        logging.info("ESC pressed. Exiting program.")
        print("Exiting program.")
        break

# ----------------------------- #
#       Release Resources
# ----------------------------- #

cap.release()
cv2.destroyAllWindows()
hands.close()
logging.info("=== Real-Time Hand Gesture Recognition Ended ===")
