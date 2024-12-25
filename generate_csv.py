import cv2
import os
import csv
import numpy as np

def segment_object(frame):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range for white color (for white background removal)
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([179, 50, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Define the range for black color (for black background removal)
    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([179, 255, 50], dtype=np.uint8)
    black_mask = cv2.inRange(hsv, lower_black, upper_black)

    # Combine both masks to remove both black and white backgrounds
    background_mask = cv2.bitwise_or(white_mask, black_mask)

    # Invert the mask to keep the object
    object_mask = cv2.bitwise_not(background_mask)

    # Apply the mask to the original image to segment the object
    segmented = cv2.bitwise_and(frame, frame, mask=object_mask)

    return segmented

# Set the directory for input images
input_directory = "./dataset/enhanced"

# Create CSV file and write header row
with open("objects.csv", mode='w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['label'] + [f'pixel_H{i}' for i in range(1, 3137)] + [f'pixel_S{i}' for i in range(1, 3137)] + [f'pixel_V{i}' for i in range(1, 3137)]  # Set column names
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Define object labels and corresponding categories
    objects = {
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6
    }

    # Process each object's folder
    for object_name, object_label in objects.items():
        object_directory = os.path.join(input_directory, object_name)
        print(f"Processing images for object {object_name}...")

        # Process each image in the object folder
        for filename in os.listdir(object_directory):
            if filename.endswith(".jpg"):
                file_path = os.path.join(object_directory, filename)
                image = cv2.imread(file_path)  # Read the image
                if image is None:
                    print(f"Could not read image: {file_path}")
                    continue

                # Segment the object from the background
                segmented = segment_object(image)

                # Ensure the segmented object is not empty
                if segmented is not None and np.any(segmented):
                    # Resize the image to 56x56 pixels
                    resized = cv2.resize(segmented, (56, 56))

                    # Flatten the HSV channels into a 1D array
                    pixel_values_H = resized[:, :, 0].flatten()  # H channel
                    pixel_values_S = resized[:, :, 1].flatten()  # S channel
                    pixel_values_V = resized[:, :, 2].flatten()  # V channel

                    # Write data to the CSV file
                    row_data = {'label': object_label}
                    for idx, (pixel_H, pixel_S, pixel_V) in enumerate(zip(pixel_values_H, pixel_values_S, pixel_values_V), start=1):
                        row_data[f'pixel_H{idx}'] = pixel_H
                        row_data[f'pixel_S{idx}'] = pixel_S
                        row_data[f'pixel_V{idx}'] = pixel_V
                    
                    writer.writerow(row_data)
                else:
                    print(f"No object detected in image: {file_path}")

print("CSV file generated successfully!")
