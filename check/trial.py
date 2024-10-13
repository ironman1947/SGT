import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import sys

# Ensure UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

# Load the pre-trained model
model = load_model(r'C:\Users\Administrator\Documents\Sign\sign_language_detection_model.h5')

# Function to preprocess the frame from webcam
def preprocess_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize to 48x48 pixels
    resized = cv2.resize(gray, (48, 48))
    # Normalize the image
    normalized = resized / 255.0
    # Reshape for model input (add batch dimension)
    reshaped = np.reshape(normalized, (1, 48, 48, 1))
    return reshaped

# Function to get the prediction label from model output
def get_label(prediction):
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
              'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
              'Y', 'Z', 'del', 'nothing', 'space']
    return labels[np.argmax(prediction)]

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Define the top-left corner of the boundary box
    top_left_x, top_left_y = 100, 100

    # Define the bottom-right corner of the boundary box
    bottom_right_x, bottom_right_y = top_left_x + 48, top_left_y + 48

    # Draw a rectangle (boundary) around the region of interest
    cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (255, 0, 0), 2)

    # Extract the region of interest for prediction
    roi = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    # Preprocess the frame
    processed_frame = preprocess_frame(roi)

    # Predict the gesture
    prediction = model.predict(processed_frame)

    # Get the predicted label
    label = get_label(prediction)

    # Print the prediction for debugging
    print(f"Prediction: {prediction}, Label: {label}")

    # Display the predicted label on the frame
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('ASL Detection', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
