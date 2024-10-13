import cv2
import mediapipe as mp
import os
import numpy as np
from tqdm import tqdm

# Initialize MediaPipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Function to extract keypoints from an image
def extract_keypoints(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract keypoints
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.append([lm.x, lm.y, lm.z])  # X, Y, Z coordinates
            keypoints = np.array(keypoints).flatten()  # Flatten to 1D array (63 values)

            # Normalize keypoints (optional, but recommended)
            keypoints = keypoints / np.linalg.norm(keypoints)  # Normalize to unit length
            return keypoints

    return np.zeros(63)  # Return zero array if no hand is detected

# Paths to your dataset (adjust paths as per your directory structure)
train_dir = r"splitdataset_Media_pipe\train"  # Replace with your train directory
val_dir = r"splitdataset_Media_pipe\val"      # Replace with your val directory

# Function to process the entire dataset folder and extract keypoints
def process_folder(folder_path):
    keypoints_data = []
    labels = []
    for label in os.listdir(folder_path):  # Each subfolder represents a gesture class
        label_dir = os.path.join(folder_path, label)
        if os.path.isdir(label_dir):
            for image_name in tqdm(os.listdir(label_dir)):
                image_path = os.path.join(label_dir, image_name)
                try:  # Try to read the image
                    image = cv2.imread(image_path)
                    if image is not None:
                        keypoints = extract_keypoints(image)
                        keypoints_data.append(keypoints)
                        labels.append(label)
                except Exception as e:  # Handle the exception if it occurs
                    print(f"Error reading image {image_path}: {e}")
    return np.array(keypoints_data), labels

# Process training and validation datasets
print("Processing train data...")
X_train, y_train = process_folder(train_dir)

print("Processing validation data...")
X_val, y_val = process_folder(val_dir)

# Convert labels to numeric
label_mapping = {label: idx for idx, label in enumerate(np.unique(y_train))}
y_train_numeric = [label_mapping[label] for label in y_train]
y_val_numeric = [label_mapping[label] for label in y_val]

# Save extracted keypoints and labels
np.save("X_train_keypoints.npy", X_train)
np.save("y_train_labels.npy", y_train_numeric)
np.save("X_val_keypoints.npy", X_val)
np.save("y_val_labels.npy", y_val_numeric)

print("Keypoints extraction and saving complete!")