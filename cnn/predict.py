import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
import os

# Load the trained model
model = load_model('object_20250103_162905.h5')

# Initialize label binarizer to map the output of the model to human-readable labels
label_binarizer = LabelBinarizer()
label_binarizer.fit([1, 2, 3, 4, 5, 6])  # Match the classes from 1 to 6

def segment_object(frame):
    """
    Function to segment the object from the background using color thresholds.
    Also performs cropping and padding logic as per the first script.
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
    
    # Draw the convex hull on the mask (white area)
    cv2.drawContours(convex_hull_mask, [hull], -1, (255), thickness=cv2.FILLED)
    
    # Create a black image of the same size as the original frame
    black_frame = np.zeros_like(frame)

    # Apply the convex hull mask to the original frame, keeping only the area inside the convex hull
    frame_with_hull = cv2.bitwise_and(frame, frame, mask=convex_hull_mask)
    
    # Combine the black image with the segmented region (inside the convex hull)
    blacked_out_frame = cv2.bitwise_or(black_frame, frame_with_hull)

    # Get the bounding rectangle of the convex hull
    x, y, w, h = cv2.boundingRect(hull)
    
    # Add 20 pixels to the bounding box (ensuring we don't go out of bounds)
    padding = 5
    x_min = max(x - padding, 0)
    y_min = max(y - padding, 0)
    x_max = min(x + w + padding, frame.shape[1])
    y_max = min(y + h + padding, frame.shape[0])
    
    # Crop the image with the padded bounding box
    cropped_image = blacked_out_frame[y_min:y_max, x_min:x_max]
    
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


def crop_to_object(image):
    """
    Crop the image to remove empty space around the segmented object.
    """
    # Convert to grayscale for easier thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Thresholding to create a binary image (objects should be white, background black)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (which should correspond to the object)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding box around the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop the image to the bounding box of the object
        cropped_image = image[y:y+h, x:x+w]
        return cropped_image
    return image

# Set up the camera
cap = cv2.VideoCapture(0)

# Check if the camera is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Path to the image file
image_path = ["./../dataset/initial_single/1/1.jpg", "./../dataset/initial_single/1/2.jpg", "./../dataset/initial_single/1/3.jpg", "./../dataset/initial_single/1/4.jpg", "./../dataset/initial_single/1/5.jpg", "./../dataset/initial_single/1/6.jpg",
              "./../dataset/initial_single/2/1.jpg", "./../dataset/initial_single/2/2.jpg", "./../dataset/initial_single/2/3.jpg", "./../dataset/initial_single/2/4.jpg", "./../dataset/initial_single/2/5.jpg", "./../dataset/initial_single/2/6.jpg",
              "./../dataset/initial_single/3/1.jpg", "./../dataset/initial_single/3/2.jpg", "./../dataset/initial_single/3/3.jpg", "./../dataset/initial_single/3/4.jpg", "./../dataset/initial_single/3/5.jpg", "./../dataset/initial_single/3/6.jpg",
              "./../dataset/initial_single/4/1.jpg", "./../dataset/initial_single/4/2.jpg", "./../dataset/initial_single/4/3.jpg", "./../dataset/initial_single/4/4.jpg", "./../dataset/initial_single/4/5.jpg", "./../dataset/initial_single/4/6.jpg",
              "./../dataset/initial_single/5/1.jpg", "./../dataset/initial_single/5/2.jpg", "./../dataset/initial_single/5/3.jpg", "./../dataset/initial_single/5/4.jpg", "./../dataset/initial_single/5/5.jpg", "./../dataset/initial_single/5/6.jpg",
              "./../dataset/initial_single/6/1.jpg", "./../dataset/initial_single/6/2.jpg", "./../dataset/initial_single/6/3.jpg", "./../dataset/initial_single/6/4.jpg", "./../dataset/initial_single/6/5.jpg", "./../dataset/initial_single/6/6.jpg"]

i = 0
while True:
    if i == len(image_path):
        i = 0
    
    # Read the image from file
    frame = cv2.imread(image_path[i])
    i += 1

    # Check if the image was read correctly
    if frame is None:
        print(f"Error: Could not read image from {image_path}")
        exit()
        
    # # Capture frame-by-frame
    # ret, frame = cap.read()
    # if not ret:
    #     print("Error: Failed to capture image.")
    #     break

    # Segment the object from the background
    padded_segmented, otsu_mask = segment_object(frame)

    # Crop the segmented image to remove empty space
    # cropped_segmented = crop_to_object(padded_segmented)

    # Ensure the cropped image is not empty
    if padded_segmented is not None and np.any(padded_segmented):
        # Resize the image to 56x56 pixels (same size as the training images)
        resized = cv2.resize(padded_segmented, (56, 56))
        cv2.imshow("Resized Image", resized)

        # # Flatten the RGB channels into a 1D array
        # rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # pixels_values_R = rgb[:, :, 0].flatten()
        # pixels_values_G = rgb[:, :, 1].flatten()
        # pixels_values_B = rgb[:, :, 2].flatten()
        # Flatten the HSV channels into a 1D array
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        pixels_values_H = hsv[:, :, 0].flatten()
        pixels_values_S = hsv[:, :, 1].flatten()
        pixels_values_V = hsv[:, :, 2].flatten()
        
        # Prepare the data for prediction (reshape to match the model input)
        x_input = np.concatenate([pixels_values_H, pixels_values_S, pixels_values_V])
        x_input = x_input.reshape(1, 56, 56, 3)  # Reshape to match the input shape of the model

        # Perform prediction
        prediction = model.predict(x_input)
        predicted_class = label_binarizer.inverse_transform(prediction)

        # Display the predicted class on the screen
        cv2.putText(frame, f"{predicted_class[0]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Live Object Prediction', frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.waitKey(200)

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
