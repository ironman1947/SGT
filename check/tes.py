import cv2
import mediapipe as mp
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Create directories for each letter
letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
dataset_path = 'media_pipe_dataset'

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
    for letter in letters:
        os.makedirs(os.path.join(dataset_path, letter))

# Function to get the current count of images in a letter's folder
def get_existing_count(letter_path):
    return len([name for name in os.listdir(letter_path) if os.path.isfile(os.path.join(letter_path, name))])

# Store count for each letter based on existing images
count_dict = {letter: get_existing_count(os.path.join(dataset_path, letter)) for letter in letters}

# Load webcam for capturing images
cap = cv2.VideoCapture(0)
print("Press the letter key (A-Z) to capture images. Press 'ESC' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR image to RGB for MediaPipe processing
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect hand landmarks
    result = hands.process(image_rgb)

    # If landmarks are found, draw them on the image
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the image
    cv2.imshow('MediaPipe Hand Detection', frame)

    # Wait for a keypress
    key = cv2.waitKey(1) & 0xFF

    # If the key is between 'A' and 'Z', capture the image
    if chr(key).upper() in letters:
        letter = chr(key).upper()
        count_dict[letter] += 1
        letter_path = os.path.join(dataset_path, letter)
        image_path = os.path.join(letter_path, f"{letter}_{count_dict[letter]}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Saved {image_path}")

    # Break the loop if 'ESC' is pressed (keycode 27)
    if key == 27:  # ESC key
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
