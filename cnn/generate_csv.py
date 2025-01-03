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
input_directory = "./../dataset/enhanced_single"

# Create CSV file and write header row
with open("objects_single_hsv.csv", mode='w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['label'] + [f'pixel_R{i}' for i in range(1, 3137)] + [f'pixel_G{i}' for i in range(1, 3137)] + [f'pixel_B{i}' for i in range(1, 3137)]  # Set column names
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

                    # # Extract RGB channels from the resized image
                    # pixel_values_R = resized[:, :, 2].flatten()  # Red channel
                    # pixel_values_G = resized[:, :, 1].flatten()  # Green channel
                    # pixel_values_B = resized[:, :, 0].flatten()  # Blue channel
                    
                    # Extract HSV channels from the resized image
                    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
                    pixel_values_H = hsv[:, :, 0].flatten()  # Hue channel
                    pixel_values_S = hsv[:, :, 1].flatten()  # Saturation channel
                    pixel_values_V = hsv[:, :, 2].flatten()  # Value channel

                    # Write data to the CSV file
                    row_data = {'label': object_label}
                    for idx, (pixel_R, pixel_G, pixel_B) in enumerate(zip(pixel_values_H, pixel_values_S, pixel_values_V), start=1):
                        row_data[f'pixel_R{idx}'] = pixel_R
                        row_data[f'pixel_G{idx}'] = pixel_G
                        row_data[f'pixel_B{idx}'] = pixel_B
                    
                    writer.writerow(row_data)
                else:
                    print(f"No object detected in image: {file_path}")

print("CSV file generated successfully!")
