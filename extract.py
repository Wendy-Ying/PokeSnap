import cv2
import numpy as np

def segment_object(frame):
    """
    Function to segment the object using HSV color space, Otsu's method, and convex hull to ensure a single combined object.
    """
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Extract the S (saturation) channel from the HSV image
    s_channel = hsv[:, :, 1]
    
    # Apply Otsu's method to find the optimal threshold value
    _, otsu_thresholded = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Perform morphological closing to remove noise and fill gaps
    kernel = np.ones((15, 15), np.uint8)
    closed_mask = cv2.morphologyEx(otsu_thresholded, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((27, 27), np.uint8)
    opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)
    
    # Inflate the mask
    kernel = np.ones((9, 9), np.uint8)
    inflated_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_DILATE, kernel)
    
    # Find contours in the closed mask
    contours, _ = cv2.findContours(inflated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create an empty mask to store the convex hull of the target object
    convex_hull_mask = np.zeros_like(inflated_mask)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest_contour)
    
    # Get the bounding rectangle of the convex hull
    x, y, w, h = cv2.boundingRect(hull)
    
    # Add 20 pixels to the bounding box (ensuring we don't go out of bounds)
    padding = 20
    x_min = max(x - padding, 0)
    y_min = max(y - padding, 0)
    x_max = min(x + w + padding, frame.shape[1])
    y_max = min(y + h + padding, frame.shape[0])
    
    # Crop the image with the padded bounding box
    cropped_image = frame[y_min:y_max, x_min:x_max]
    
    # Make the cropped image square by adding black padding
    height, width = cropped_image.shape[:2]
    max_dim = max(height, width)
    
    # Create a new black image (padding with black pixels)
    padded_image = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    
    # Place the cropped image in the center of the black image
    y_offset = (max_dim - height) // 2
    x_offset = (max_dim - width) // 2
    padded_image[y_offset:y_offset + height, x_offset:x_offset + width] = cropped_image
    
    return padded_image, otsu_thresholded

# Capture a frame from the camera
cap = cv2.VideoCapture(0)
# Check if the camera is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize frame counter
frame_count = 0

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Increment frame counter
    frame_count += 1

    # Process every 10th frame
    if frame_count % 10 == 0:
        # Segment the object from the background using HSV + Otsu's method
        padded_image, otsu_mask = segment_object(frame)

        # Display the padded image
        cv2.imshow('Padded Image', padded_image)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
