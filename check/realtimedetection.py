from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import cv2
import numpy as np
import json
import keras
import sys

# Ensure the default encoding is UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Register the Sequential class
@keras.saving.register_keras_serializable()
class CustomSequential(Sequential):
    pass

# Load the model JSON
with open('sign_language_detection_model_l.json', 'r') as json_file:
    model_json = json_file.read()

# Modify the JSON to remove batch_input_shape from Conv2D layers
model_config = json.loads(model_json)

for layer in model_config['config']['layers']:
    if 'class_name' in layer and layer['class_name'] == 'Conv2D':
        if 'batch_input_shape' in layer['config']:
            del layer['config']['batch_input_shape']

modified_model_json = json.dumps(model_config)

# Load the modified model JSON with custom_objects
model = model_from_json(modified_model_json, custom_objects={'Sequential': CustomSequential})

# Compile the model (if necessary)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Model loaded and compiled successfully!")

model.load_weights("sign_language_detection_model_l.h5")

# Function to preprocess the image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Define the labels
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'blank']

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define the region of interest
    cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
    crop_frame = frame[40:300, 0:300]
    crop_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
    crop_frame = cv2.resize(crop_frame, (48, 48))
    crop_frame = extract_features(crop_frame)

    # Predict the gesture
    pred = model.predict(crop_frame)
    prediction_label = labels[pred.argmax()]
    accuracy = "{:.2f}".format(np.max(pred) * 100)

    # Display the predicted label and accuracy
    cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
    display_text = f"{prediction_label}  {accuracy}%" if prediction_label != 'blank' else " "
    cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("output", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
