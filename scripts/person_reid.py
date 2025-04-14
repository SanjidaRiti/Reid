import cv2
from detect_and_track import detect_and_track_person
from feature_extraction import extract_feature
from utils import cosine_similarity, load_model
import numpy as np
import torch

# Load the pre-trained ReID model
model = load_model()

# Video input path
video_path = 'data/walking2.mp4'
cap = cv2.VideoCapture(video_path)

# Initialize a list to store the previous frame's features (for re-identification comparison)
previous_features = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect people in the current frame
    bboxes, confidences = detect_and_track_person(frame)  # Returns bounding boxes and confidences
    
    for bbox in bboxes:
        # Crop the person out of the frame (simple example, assumes you have a bbox)
        x1, y1, x2, y2 = bbox
        person_crop = frame[y1:y2, x1:x2]
        
        # Extract the feature of the person
        feature = extract_feature(person_crop, model)
        
        # Compare with previous features for re-identification (use cosine similarity)
        if previous_features:
            similarities = [cosine_similarity(feature, prev_feat) for prev_feat in previous_features]
            print(f"Similarities: {similarities}")
        
        # Add the current feature to the list of previous features for future comparison
        previous_features.append(feature)
        
    # Optionally, display the frame with detections (for debugging)
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
