import cv2

def detect_and_track_person(frame):
    """
    Detect and track persons in the given frame.
    """
    # Example using OpenCV's HOG descriptor for pedestrian detection
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # Detect people in the frame
    bboxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(16, 16), scale=1.05)
    
    # Filter out weak detections based on confidence score (weights)
    bboxes = [bbox for bbox, weight in zip(bboxes, weights) if weight > 0.5]  # Confidence threshold
    
    return bboxes, weights
