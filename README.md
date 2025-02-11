# Sign Language Recognition

This project utilizes deep learning to recognize sign language gestures from images. It leverages a custom CNN for gesture recognition and a pre-trained MobileNet model for feature extraction. 

---

## Table of Contents
- [About the Project](#about-the-project)
- [Features](#features)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

---

## About the Project
The **Sign Language Recognition** project is designed to:
- Identify and classify hand gestures representing sign language characters.
- Utilize a pre-trained **MobileNet** model for feature extraction.
- Deploy a custom CNN model for final classification.

The project enables efficient recognition of sign language gestures, making communication more accessible for individuals with hearing or speech impairments.

---

## Features
- **Data Preprocessing**: Processes input images for compatibility with the trained models.
- **Feature Extraction**: Uses the pre-trained **MobileNet** model for efficient feature extraction.
- **Gesture Classification**: Employs a custom CNN to classify sign language gestures.
- **Real-time Recognition**: Recognizes and classifies gestures from images.

---

## Getting Started

Follow these steps to set up and run the project locally:

### Prerequisites
- Python 3.10+
- TensorFlow/Keras
- OpenCV
- NumPy

---

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-link>
   cd Sign_Language_Recognition
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the pre-trained model is in the project directory:
   - `sign_language_model.h5`: Pre-trained model for sign language recognition.

---

## Usage

To use the application, follow these steps:

1. Run the Python script to load the model and test it on sample images:
   ```bash
   python test.py --image path/to/image.jpg
   ```
2. The script will process the image and display the recognized sign language gesture.

---

## Project Structure
The project files are organized as follows:
```
Sign_Language_Recognition/
├── Sign_Language_model.ipynb  # Notebook for training the sign language recognition model
├── test.py                    # Script for loading the model and making predictions
├── sign_language_model.h5     # Pre-trained model for sign language recognition
├── readme                     # Project documentation
```

---

This project aims to facilitate communication for the hearing-impaired community by providing an efficient and accurate sign language recognition system.

