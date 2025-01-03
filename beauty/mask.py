import cv2
import numpy as np

class ObjectSegmentation:
    def __init__(self, padding=50, alpha=0.8):
        self.padding = padding  # Padding around the object in the bounding box
        self.alpha = alpha  # Alpha value for blending

    def segment_object(self, frame, bg):
        """
        Function to segment the object from the background using color thresholds.
        This version removes background logic and uses a black background instead.
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
        
        # Apply the convex hull mask to the original frame, keeping only the area inside the convex hull
        frame_with_hull = cv2.bitwise_and(frame, frame, mask=convex_hull_mask)
        
        # Create a black background and add the segmented object to it
        black_background = np.zeros_like(frame)
        final_image = cv2.add(black_background, frame_with_hull)

        # Get the bounding rectangle of the convex hull
        x, y, w, h = cv2.boundingRect(hull)
        
        # Add padding to the bounding box (ensuring we don't go out of bounds)
        x_min = max(x - self.padding, 0)
        y_min = max(y - self.padding, 0)
        x_max = min(x + w + self.padding, frame.shape[1])
        y_max = min(y + h + self.padding, frame.shape[0])
        
        # Crop the image with the padded bounding box
        cropped_image = final_image[y_min:y_max, x_min:x_max]
        
        # Make the cropped image square by adding padding
        height, width = cropped_image.shape[:2]
        max_dim = max(height, width)
        
        # Create a new image with padding, initially set to a black background
        padded_image = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)

        # Place the cropped image in the center of the padded image
        y_offset = (max_dim - height) // 2
        x_offset = (max_dim - width) // 2
        padded_image[y_offset:y_offset + height, x_offset:x_offset + width] = cropped_image

        # Resize the image to 512x512
        padded_image = cv2.resize(padded_image, (512, 512))
        
        # Create mask for blending
        mask = cv2.cvtColor(padded_image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        
        # Ensure the mask has the correct size and type
        mask = mask.astype(np.uint8)  # Ensure the mask is of type CV_8U
        
        # Invert mask
        mask_inv = cv2.bitwise_not(mask)
        
        # Apply mask to background
        masked_bg = cv2.bitwise_and(bg, bg, mask=mask_inv)
        
        # Apply mask to padded image
        masked_padded_image = cv2.bitwise_and(padded_image, padded_image, mask=mask)
        
        # Combine the masked background and the masked padded image
        final_image = cv2.add(masked_padded_image, masked_bg)
        
        # Apply Gaussian blur to the final image to smooth the edges
        blurred_final_image = cv2.GaussianBlur(final_image, (3, 3), 0)
        
        # Convert the images to float for alpha blending
        bg_float = bg.astype(np.float32) / 255.0
        fg_float = blurred_final_image.astype(np.float32) / 255.0
        
        # Perform alpha blending
        blended_image = (bg_float * (1 - self.alpha)) + (fg_float * self.alpha)
        
        # Convert back to uint8
        blended_image = (blended_image * 255).astype(np.uint8)
        
        return blended_image

# Example usage
if __name__ == '__main__':
    image_path = "./../dataset/test/3.jpg"
    image = cv2.imread(image_path)
    bg_path = "bg.jpg"
    bg = cv2.imread(bg_path)
    bg = cv2.resize(bg, (512, 512))

    # Initialize the ObjectSegmentation class
    segmenter = ObjectSegmentation(padding=50, alpha=0.8)
    
    # Segment and beautify the image
    segmented_image = segmenter.segment_object(image, bg)
    
    # Resize and display the final result
    resized_image = cv2.resize(segmented_image, (512, 512))
    cv2.imshow("Beautified Image", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
