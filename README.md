# Sign Language Recognition for Mute People

## Overview

This project is designed to assist mute individuals by translating hand gestures into text and audio output in real time. The system relies on a custom Deep Neural Network (DNN) model that captures hand movements and accurately classifies them into specific commands or letters. This project was developed by a team of five members and received 3rd place in a national project presentation competition.

## Key Features

- **Real-Time Gesture Recognition**: The model uses MediaPipe to capture and process hand gestures in real time.
- **Custom Data Input Phase**: Unlike traditional sign language recognition systems, this model has a unique data input phase that leverages both custom labels and real-time detection, enhancing the system's accuracy and usability.
- **Audio Feedback for Users**: Outputs include synthesized voice prompts for enhanced accessibility.

## Project Structure

### Data Input (`Data_input.py`) 
- Utilizes the MediaPipe library to capture hand keypoints and save them for training.
- Supports labels for the alphabet (A-Z) along with custom commands such as "Confirm," "Space," and "Speak."
- Keypoint data is stored in a structured dataset directory, ensuring compatibility with the model training phase.

### Model Training (`ltrain.py`)
- The model is trained on a dataset split into training and validation folders.
- Employs a CNN-based architecture within a Sequential model to recognize gesture patterns.
- Includes image augmentation for robust training and optimized accuracy.
- Uses TensorFlow and Keras libraries for model development.

### Real-Time Sentence Generation (`Real_time_sentence.py`)
- Detects gestures and translates them into real-time sentences, offering continuous feedback for users.
- Includes pyttsx3 text-to-speech functionality, which provides verbal output for recognized gestures.
- Utilizes UTF-8 encoding and logging features to track system outputs and handle Unicode characters effectively.

## Model Architecture

Our model is based on a Deep Neural Network (DNN) tailored to hand gesture recognition tasks. Key components include:

- **Convolutional Layers** for capturing spatial patterns of gestures.
- **Max Pooling** to reduce dimensionality and improve generalization.
- **Dropout Layers** for preventing overfitting.

The model is trained with a categorical cross-entropy loss function, and an Adam optimizer ensures efficient learning.

## Installation

1. Clone this repository:
   ```bash
   git clone <https://github.com/ironman1947/SGT.git>
2. Install the required libraries:
    ```bash
   pip install -r requirements.txt

3. Run the Data_input.py script to prepare the dataset with MediaPipe:
    ```bash
     python Data_input.py

## Usage
1. Train the model with:
    ```bash
   python ltrain.py

2. Start the real-time gesture detection:
    ```bash
     python Real_time_sentence.py

## Future Enhancements
1. **Enhanced Vocabulary**: Expanding the model’s vocabulary beyond the current alphabet and commands.
2. **Improved Detection Accuracy**: Integrating additional data sources or sensors to improve accuracy and usability.

### Contributors
This project was developed by a team of 5 members and presented at a national competition, where it was awarded 3rd place.
- **Om Pradip Chougule** - Project Lead, Model Development
- **Jija Bhosale** - Data Processing and Preprocessing
- **Nandini Patil** - Training and Model Tuning
- **Satyajeet Kasabekar** - Real-Time System Integration
- **Viraj Musale** - Testing and Evaluation
