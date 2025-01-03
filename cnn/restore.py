import csv
import numpy as np
import os
import random
import cv2

# Set CSV file path
csv_file = "objects_single_rgb.csv"

# Set the directory for saving restored images
output_directory = "./restored_images_rgb"

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Read CSV and restore images
def restore_images_from_csv(csv_file, num_images=100):
    # Store image data read from CSV
    images_data = []

    # Open the CSV file and read the data
    with open(csv_file, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        print(reader.fieldnames)

        # Loop through each row in the CSV
        for row in reader:
            label = int(row['label'])
            pixel_values_R = [int(row[f'pixel_R{i}']) for i in range(1, 3137)]
            pixel_values_G = [int(row[f'pixel_G{i}']) for i in range(1, 3137)]
            pixel_values_B = [int(row[f'pixel_B{i}']) for i in range(1, 3137)]

            # Reshape the R, G, B channels into 56x56 images
            r_channel = np.array(pixel_values_R, dtype=np.uint8).reshape((56, 56))
            g_channel = np.array(pixel_values_G, dtype=np.uint8).reshape((56, 56))
            b_channel = np.array(pixel_values_B, dtype=np.uint8).reshape((56, 56))

            # Merge the RGB channels into a single image
            restored_image = cv2.merge([b_channel, g_channel, r_channel])  # OpenCV uses BGR by default

            # Save the image data and label
            images_data.append((restored_image, label))

    # Randomly select num_images images
    random_images = random.sample(images_data, num_images)

    # Restore and save the images
    for idx, (image, label) in enumerate(random_images):
        # Save the restored image
        output_filename = os.path.join(output_directory, f"restored_image_{idx+1}_label_{label}.png")
        cv2.imwrite(output_filename, image)

    print(f"{num_images} images have been restored and saved to {output_directory}.")

# Call the function to restore and save images
restore_images_from_csv(csv_file)
