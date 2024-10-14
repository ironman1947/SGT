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
import pyttsx3
import time

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
MODEL_PATH = 'hand_gesture_model (1).h5'
LABEL_ENCODER_PATH = 'label_encoder.pkl'
SCALER_PATH = 'scaler.pkl'

# Load the trained model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logging.info(f"Loaded model from '{MODEL_PATH}' successfully.")
except Exception as e:
    logging.error(f"Failed to load model from '{MODEL_PATH}': {e}")
    print(f"Error: Failed to load model from '{MODEL_PATH}'. Check logs for details.")
    sys.exit(1)

# Load the Label Encoder and Scaler
try:
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    logging.info(f"Loaded label encoder from '{LABEL_ENCODER_PATH}' and scaler from '{SCALER_PATH}' successfully.")
except Exception as e:
    logging.error(f"Failed to load label encoder or scaler: {e}")
    print("Error: Failed to load label encoder or scaler. Check logs for details.")
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
#       Initialize TTS Engine
# ----------------------------- #

# Initialize the TTS engine
tts_engine = pyttsx3.init()

# Optionally, set properties like voice, rate, and volume
tts_engine.setProperty('rate', 150)    # Speech rate
tts_engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)

# ----------------------------- #
#       Initialize State Variables
# ----------------------------- #

current_letter = None
word = ""
sentence = ""

# Debounce parameters
DEFAULT_DEBOUNCE_TIME = 0.5  # seconds for general gestures
DEBOUNCE_TIMES = {
    'confirm': 1.0,  # seconds cooldown for 'confirm' gesture
    'space': 1.0,    # seconds cooldown for 'space' gesture
    'speak': 1.5     # seconds cooldown for 'speak' gesture
}
last_gesture_time = {}  # Dictionary to store the last detection time for each gesture

# ----------------------------- #
#       Prediction and TTS Functions
# ----------------------------- #

def is_debounced(gesture_label, debounce_time=DEFAULT_DEBOUNCE_TIME):
    """
    Checks if the gesture can be registered based on the debounce time.
    
    Args:
        gesture_label (str): The label of the detected gesture.
        debounce_time (float): The debounce time in seconds.
    
    Returns:
        bool: True if the gesture can be registered, False otherwise.
    """
    current_time = time.time()
    if gesture_label in last_gesture_time:
        elapsed_time = current_time - last_gesture_time[gesture_label]
        if elapsed_time < debounce_time:
            return False
    last_gesture_time[gesture_label] = current_time
    return True

def speak_sentence(sentence):
    """
    Converts the given sentence to speech.
    
    Args:
        sentence (str): The sentence to be spoken.
    """
    try:
        tts_engine.say(sentence)
        tts_engine.runAndWait()
        logging.info(f"Spoken Sentence: {sentence}")
    except Exception as e:
        logging.error(f"Error during speech synthesis: {e}")

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
            predicted_gesture = predict_gesture(keypoints)

            # Debounce gesture detection
            gesture_label = predicted_gesture.lower()
            debounce_time = DEBOUNCE_TIMES.get(gesture_label, DEFAULT_DEBOUNCE_TIME)
            if is_debounced(gesture_label, debounce_time):
                logging.info(f"Detected Gesture: {predicted_gesture}")

                if predicted_gesture.upper() in [chr(i) for i in range(65, 91)]:  # A-Z
                    current_letter = predicted_gesture.upper()
                    # Display detected letter
                    cv2.putText(
                        frame,
                        f"Detected Letter: {current_letter}",
                        (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA
                    )
                elif gesture_label == 'confirm':
                    if current_letter:
                        word += current_letter
                        current_letter = None
                        logging.info(f"Added Letter: {word}")
                elif gesture_label == 'space':
                    sentence += word + ' '
                    word = ""
                    logging.info(f"Added Space. Current Sentence: '{sentence}'")
                elif gesture_label == 'speak':
                    full_sentence = sentence + word
                    if full_sentence.strip():
                        speak_sentence(full_sentence)
                        logging.info(f"Spoken Sentence: '{full_sentence}'")
                        # Optionally, reset the sentence after speaking
                        sentence = ""
                        word = ""
                else:
                    logging.warning(f"Unknown Gesture: {predicted_gesture}")

    # Display the current word and sentence
    cv2.putText(
        frame,
        f"Word: {word}",
        (10, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        frame,
        f"Sentence: {sentence}",
        (10, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

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
tts_engine.stop()
logging.info("=== Real-Time Hand Gesture Recognition Ended ===")
