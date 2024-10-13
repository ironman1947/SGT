import cv2
import mediapipe as mp
import os
import time

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

# Path to dataset
dataset_path = 'media_pipe_dataset'

# Create directories for each label if they don't exist
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
    for label in all_labels:
        label_dir = os.path.join(dataset_path, label)
        os.makedirs(label_dir, exist_ok=True)
        print(f"Created directory: {label_dir}")

# Function to get the current count of images in a label's folder
def get_existing_count(label_path):
    return len([
        name for name in os.listdir(label_path)
        if os.path.isfile(os.path.join(label_path, name))
    ])

# Initialize count dictionary
count_dict = {}
for label in all_labels:
    label_path = os.path.join(dataset_path, label)
    count = get_existing_count(label_path)
    count_dict[label] = count
    print(f"Initial count for '{label}': {count}")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("\n=== Image Capture Instructions ===")
print("Press the corresponding key to capture images:")
print(" - Letter keys (A-Z) for alphabet letters.")
print(" - Number key '1' for 'Confirm'")
print(" - Number key '2' for 'Space'")
print(" - Number key '3' for 'Speak'")
print("Press 'ESC' to quit.\n")

# Initialize debounce variables
last_capture_time = 0
capture_delay = 0.3  # seconds

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Note: Removed frame flipping to maintain natural orientation
    # frame = cv2.flip(frame, 1)

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
        "Press A-Z or 1-3 to capture. ESC to quit.",
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
    if key in key_to_label:
        if (current_time - last_capture_time) > capture_delay:
            label = key_to_label[key]
            count_dict[label] += 1
            label_path = os.path.join(dataset_path, label)
            image_filename = f"{label}_{count_dict[label]}.jpg"
            image_path = os.path.join(label_path, image_filename)
            cv2.imwrite(image_path, frame)
            print(f"Saved: {image_path}")
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
