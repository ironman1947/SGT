import cv2
import mediapipe as mp
import os
import time
import csv

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Define labels: Letters A-Z and custom labels
letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
additional_labels = {
    '1': 'Confirm',
    '2': 'Space',
    '3': 'Speak'
}
all_labels = letters + list(additional_labels.values())

# Define key-to-label mapping
key_to_label = {ord(k): v for k, v in additional_labels.items()}
for letter in letters:
    key_to_label[ord(letter.lower())] = letter  # Lowercase keys
    key_to_label[ord(letter.upper())] = letter  # Uppercase keys

# Path to dataset (for saving keypoints)
dataset_path = 'media_pipe_keypoints_dataset'

# Create directories for each label if they don't exist
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
    for label in all_labels:
        label_dir = os.path.join(dataset_path, label)
        os.makedirs(label_dir, exist_ok=True)
        print(f"Created directory: {label_dir}")

# Initialize count dictionary
count_dict = {label: 0 for label in all_labels}

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("\n=== Keypoint Capture Instructions ===")
print("Press the corresponding key to capture keypoints:")
print(" - Letter keys (A-Z) for alphabet letters.")
print(" - Number key '1' for 'Confirm'")
print(" - Number key '2' for 'Space'")
print(" - Number key '3' for 'Speak'")
print("Press 'ESC' to quit.\n")

# Initialize debounce variables
last_capture_time = 0
capture_delay = 0.3  # seconds

def save_keypoints_to_csv(label, keypoints, count):
    label_path = os.path.join(dataset_path, label)
    csv_filename = os.path.join(label_path, f"{label}_{count}.csv")

    # Save keypoints to CSV
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Landmark', 'X', 'Y', 'Z', 'Visibility'])  # CSV headers
        for idx, landmark in enumerate(keypoints.landmark):
            writer.writerow([idx, landmark.x, landmark.y, landmark.z, landmark.visibility])

    print(f"Saved keypoints: {csv_filename}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert BGR image to RGB for MediaPipe processing
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect hand landmarks
    result = hands.process(image_rgb)

    # Draw hand landmarks if detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    # Display instructions on the frame
    cv2.putText(
        frame,
        "Press A-Z or 1-3 to capture keypoints. ESC to quit.",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Show the frame
    cv2.imshow('MediaPipe Hand Detection', frame)

    # Wait for a keypress for 1ms
    key = cv2.waitKey(1) & 0xFF
    current_time = time.time()

    # Check if the pressed key corresponds to any label
    if key in key_to_label and result.multi_hand_landmarks:
        if (current_time - last_capture_time) > capture_delay:
            label = key_to_label[key]
            count_dict[label] += 1

            # Save the detected hand keypoints for the current frame
            for hand_landmarks in result.multi_hand_landmarks:
                save_keypoints_to_csv(label, hand_landmarks, count_dict[label])

            last_capture_time = current_time
        else:
            print("Capture skipped to prevent duplicate.")

    # Exit if ESC is pressed
    if key == 27:
        print("Exiting program.")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
