import cv2
import os
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import tensorflow as tf

# Check if TensorFlow Model Exists
model_path = "Model/model.h5"
labels_path = "Model/labels.txt"

if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found!")
    exit()

if not os.path.exists(labels_path):
    print(f"Error: Labels file '{labels_path}' not found!")
    exit()

# Read labels
with open(labels_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize video capture and hand detector
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

detector = HandDetector(maxHands=1)

# Load the model with custom_objects to fix the activation function
try:
    model = tf.keras.models.load_model(model_path, custom_objects={
        'softmax_v2': tf.nn.softmax  # Replace 'softmax_v2' with 'tf.nn.softmax'
    })
except Exception as e:
    print(f"Failed to load model: {str(e)}")
    exit()

# Parameters
offset = 20
img_size = 224  # Standard input size for many CNN models

def preprocess_hand_image(img_crop, target_size):
    """Preprocess the cropped hand image for model input."""
    if img_crop.size == 0:
        return None
    
    # Create a white background
    white_bg = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
    
    # Maintain aspect ratio
    h, w = img_crop.shape[:2]
    if w == 0 or h == 0:
        return None
    
    aspect_ratio = h / w
    
    if aspect_ratio > 1:
        new_height = target_size
        new_width = int(target_size / aspect_ratio)
    else:
        new_width = target_size
        new_height = int(target_size * aspect_ratio)
    
    # Resize the image
    img_resized = cv2.resize(img_crop, (new_width, new_height))
    
    # Center the image on white background
    y_start = (target_size - new_height) // 2
    x_start = (target_size - new_width) // 2
    white_bg[y_start:y_start + new_height, x_start:x_start + new_width] = img_resized
    
    # Normalize the image
    processed_img = white_bg.astype(np.float32) / 255.0
    
    # Add batch dimension
    return np.expand_dims(processed_img, axis=0)

# Previous imports and setup remain the same...

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Create a margin at the top of the frame for text
    margin = 60
    frame_with_margin = np.concatenate([np.zeros((margin, frame.shape[1], 3), dtype=np.uint8), frame])

    # Detect hands in the frame
    hands, frame_with_margin[margin:] = detector.findHands(frame)

    if hands and len(hands) > 0:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        # Adjust y coordinate to account for margin
        y += margin

        # Crop the hand region with padding
        x1, y1 = max(0, x - offset), max(margin, y - offset)
        x2, y2 = min(frame_with_margin.shape[1], x + w + offset), min(frame_with_margin.shape[0], y + h + offset)
        img_crop = frame_with_margin[y1:y2, x1:x2]

        if img_crop.size == 0:
            continue

        processed_img = preprocess_hand_image(img_crop, img_size)
        
        if processed_img is not None:
            try:
                predictions = model.predict(processed_img, verbose=0)
                predictions = tf.nn.softmax(predictions).numpy()
                index = np.argmax(predictions[0])
                confidence = predictions[0][index]
                
                # Draw prediction text in the margin area
                prediction_text = f"Prediction: {labels[index]}"
                cv2.putText(frame_with_margin, prediction_text, 
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (255, 255, 255), 2)
                
                cv2.imshow('Processed Hand', (processed_img[0] * 255).astype(np.uint8))
                
            except Exception as e:
                print(f"Prediction error: {str(e)}")

    # Display the frame with margin
    cv2.imshow('Frame', frame_with_margin)

    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Frame', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()