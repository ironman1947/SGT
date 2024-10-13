import cv2
import numpy as np
import mediapipe as mp
from keras.models import model_from_json
import json
import pyttsx3
import sys

# Set UTF-8 encoding for standard output
sys.stdout.reconfigure(encoding='utf-8')

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load the model JSON
with open('sign_language_detection_model.json', 'r') as json_file:
    model_json = json_file.read()

# Load the model from JSON
model = model_from_json(model_json)

# Load the model weights
model.load_weights("sign_language_detection_model.h5")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Function to extract features from hand landmarks
def extract_features(landmarks):
    # Extract the x, y coordinates from the landmarks
    features = np.array([[landmark.x, landmark.y] for landmark in landmarks.landmark]).flatten()
    
    # Ensure that we have exactly 63 features
    if features.size < 63:
        # Pad the features with zeros if less than 63
        padded_features = np.pad(features, (0, 63 - features.size), 'constant')
    else:
        # Take only the first 63 features if more than 63
        padded_features = features[:63]
    
    return padded_features

# Define the labels including the new actions
labels = ['speak', 'confirm', 'space', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'blank']

# Open the webcam
cap = cv2.VideoCapture(0)

# Sentence accumulation
sentence = ""
current_letter = ""  # Store the current predicted letter

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame with MediaPipe Hands
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Initialize prediction_label and accuracy to avoid NameError
    prediction_label = "unknown"
    accuracy = "0.00"

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the features from the hand landmarks
            processed_frame = extract_features(hand_landmarks)

            # Reshape for the model input (1, 63)
            processed_frame = processed_frame.reshape(1, 63)  # Reshape to match model input shape

            # Predict the gesture
            pred = model.predict(processed_frame)
            prediction_index = pred.argmax()  # Get the index of the highest probability
            prediction_label = labels[prediction_index]
            accuracy = "{:.2f}".format(np.max(pred) * 100)  # Calculate accuracy

            # Optional: Draw landmarks on the frame for visualization
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Display the predicted action and accuracy
    cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
    display_text = f"{prediction_label}  {accuracy}%" if prediction_label != 'blank' else " "
    cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # Display the accumulated sentence
    cv2.putText(frame, sentence, (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Sign Language to Sentence", frame)

    # Check for keypresses
    key = cv2.waitKey(1) & 0xFF

    # Perform actions based on detected sign
    if prediction_label == 'speak':
        if sentence:
            engine.say(sentence)  # Convert text to speech
            engine.runAndWait()
            print(f"Speaking: {sentence}")
            sentence = ""  # Reset sentence after speaking

    elif prediction_label == 'confirm':
        if current_letter:  # Only add if there's a current letter
            sentence += current_letter
            current_letter = ""  # Reset current_letter
        print(f"Confirmed: {sentence}")

    elif prediction_label == 'space':
        if len(sentence) < 50:  # Modify max sentence length as needed
            sentence += " "
            print("Space added to the sentence")

    # Update current_letter (if it's a letter)
    if prediction_label not in ['speak', 'confirm', 'space', 'blank', 'unknown'] and len(sentence) < 50:
        current_letter = prediction_label  # Update the current letter

    # Break the loop if 'q' is pressed
    if key == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()