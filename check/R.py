import cv2
import numpy as np
import mediapipe as mp
from keras.models import model_from_json
import json
import pyttsx3
import sys

sys.stdout.reconfigure(encoding='utf-8')  # Set UTF-8 for standard output
# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load the model JSON
with open('sign_d.json', 'r') as json_file:
    model_json = json_file.read()

# Load the model
model = model_from_json(model_json)

# Load the model weights
model.load_weights("sign_d.weights.h5")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Function to extract features from hand landmarks
def extract_features(landmarks):
    features = np.array([[landmark.x, landmark.y] for landmark in landmarks.landmark]).flatten()
    padded_features = np.pad(features, (0, 63 - features.size), 'constant') if features.size < 63 else features[:63]
    return padded_features

# Define the labels (make sure this matches your training data)
labels = ['speak', 'confirm', 'space', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'blank']  

# Open the webcam
cap = cv2.VideoCapture(0)

# Sentence accumulation
sentence = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame (optional)
    frame = cv2.resize(frame, (640, 480))

    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    prediction_label = "unknown"
    accuracy = "0.00"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            processed_frame = extract_features(hand_landmarks)
            processed_frame = processed_frame.reshape(1, 63)  # Reshape for model input

            pred = model.predict(processed_frame)
            prediction_index = pred.argmax()  # Get the index of the highest probability
            prediction_label = labels[prediction_index]
            accuracy = "{:.2f}".format(np.max(pred) * 100)

            # Visualize landmarks (optional)
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the predicted action and accuracy
    cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
    display_text = f"{prediction_label}  {accuracy}%" if prediction_label != 'blank' else " "
    cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # Display the accumulated sentence
    cv2.putText(frame, sentence, (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Sign Language to Sentence", frame)

    key = cv2.waitKey(1) & 0xFF

    # Perform actions based on detected sign
    if prediction_label == 'speak':
        if sentence:
            engine.say(sentence)
            engine.runAndWait()
            print(f"Speaking: {sentence}")
            sentence = ""  # Reset sentence after speaking

    elif prediction_label == 'confirm':
        print(f"Confirmed: {sentence}")

    elif prediction_label == 'space':
        if len(sentence) < 50:
            sentence += " "

    # Add the predicted letter to the sentence (if not a control sign)
    if prediction_label not in ['speak', 'confirm', 'space', 'blank'] and len(sentence) < 50:
        sentence += prediction_label

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()