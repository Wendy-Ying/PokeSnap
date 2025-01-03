import cv2
import os
import numpy as np
import logging

# Initialize logging for better tracking of the process
logging.basicConfig(level=logging.INFO)

def segment_object(frame, lower_white, upper_white, lower_black, upper_black):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for white and black colors
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    black_mask = cv2.inRange(hsv, lower_black, upper_black)

    # Combine the masks and invert to segment the object
    background_mask = cv2.bitwise_or(white_mask, black_mask)
    object_mask = cv2.bitwise_not(background_mask)

    # Apply the mask to the image to isolate the object
    segmented = cv2.bitwise_and(frame, frame, mask=object_mask)
    
    return segmented

def process_images(input_directory, output_directory, gesture_labels):
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    # Define HSV thresholds for black and white background removal
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([179, 50, 255], dtype=np.uint8)
    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([179, 255, 50], dtype=np.uint8)

    # Process each gesture's images
    for gesture_name, gesture_label in gesture_labels.items():
        gesture_directory = os.path.join(input_directory, gesture_name)
        logging.info(f"Processing images for gesture {gesture_name}...")

        # Check if the gesture directory exists
        if not os.path.isdir(gesture_directory):
            logging.warning(f"Gesture directory {gesture_directory} does not exist!")
            continue
        
        # Process each image in the gesture directory
        for filename in os.listdir(gesture_directory):
            if filename.endswith(".jpg"):
                file_path = os.path.join(gesture_directory, filename)

                try:
                    # Read and process the image
                    image = cv2.imread(file_path)
                    if image is None:
                        raise ValueError(f"Could not read image: {file_path}")
                    
                    # Segment the object in the image
                    segmented_image = segment_object(image, lower_white, upper_white, lower_black, upper_black)
                    
                    # Resize the image to 224x224
                    resized_image = cv2.resize(segmented_image, (224, 224))
                    
                    # Create output path for the processed image
                    output_path = os.path.join(output_directory, gesture_name)
                    os.makedirs(output_path, exist_ok=True)
                    processed_image_path = os.path.join(output_path, filename)
                    
                    # Save the processed image (convert back to BGR for saving)
                    cv2.imwrite(processed_image_path, resized_image)

                except Exception as e:
                    logging.error(f"Error processing {file_path}: {str(e)}")

def main():
    # Directories and gestures
    input_directory = "./../dataset/enhanced_single"
    output_directory = "./../dataset/preprocessed"
    
    objects = {
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6
    }

    # Process the images
    process_images(input_directory, output_directory, objects)

    logging.info("Image processing completed successfully.")

if __name__ == "__main__":
    main()
