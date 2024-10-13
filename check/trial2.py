import numpy as np
from tensorflow.keras.models import load_model
import cv2
import sys

# Ensure UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

# Load the pre-trained model
model = load_model(r"C:\Users\Administrator\Documents\Sign\sign_language_detection_model.h5")

# Function to get the letter from the model's prediction
def getletters(result):
    classLabels = {
0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F',6:'G', 7:'M', 8:'N', 9:'S', 10:'T', 11:'blank'
    }
    res = np.argmax(result)
    return classLabels.get(res, "error")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define the region of interest
    roi = frame[40:300, 0:300]
    cv2.imshow('ROI', roi)

    # Preprocess the ROI
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (48, 48), interpolation=cv2.INTER_AREA)
    roi = roi.reshape(1, 48, 48, 1)

    # Make prediction
    label = "No prediction"
    try:
        prediction = model.predict(roi)
        label = getletters(prediction)
    except Exception as e:
        print(f"Prediction error: {e}")

    # Display the predicted label
    frame_copy = frame.copy()
    cv2.rectangle(frame_copy, (0, 0), (300, 40), (0, 165, 255), -1)
    cv2.putText(frame_copy, label, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Frame', frame_copy)

    # Exit if 'Enter' key is pressed
    if cv2.waitKey(1) == 13:
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
