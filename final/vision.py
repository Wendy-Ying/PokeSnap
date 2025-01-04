import cv2
import numpy as np
import joblib

class ImageProcessor:
    def __init__(self, resize_dim=(56, 56)):
        self.resize_dim = resize_dim
    
    def segment_object(self, frame):
        # segment ROI
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]
        
        _, otsu_thresholded = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = np.ones((15, 15), np.uint8)
        closed_mask = cv2.morphologyEx(otsu_thresholded, cv2.MORPH_CLOSE, kernel)
        
        kernel = np.ones((27, 27), np.uint8)
        opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)
        
        kernel = np.ones((9, 9), np.uint8)
        inflated_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_DILATE, kernel)
        
        contours, _ = cv2.findContours(inflated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        convex_hull_mask = np.zeros_like(inflated_mask)
        largest_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest_contour)
        
        cv2.drawContours(convex_hull_mask, [hull], -1, (255), thickness=cv2.FILLED)
        
        black_frame = np.zeros_like(frame)
        frame_with_hull = cv2.bitwise_and(frame, frame, mask=convex_hull_mask)
        blacked_out_frame = cv2.bitwise_or(black_frame, frame_with_hull)
        
        x, y, w, h = cv2.boundingRect(hull)
        x_min, y_min = max(x - 5, 0), max(y - 5, 0)  # padding stays at 5
        x_max, y_max = min(x + w + 5, frame.shape[1]), min(y + h + 5, frame.shape[0])
        
        cropped_image = blacked_out_frame[y_min:y_max, x_min:x_max]
        
        height, width = cropped_image.shape[:2]
        max_dim = max(height, width)
        padded_image = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        
        y_offset, x_offset = (max_dim - height) // 2, (max_dim - width) // 2
        padded_image[y_offset:y_offset + height, x_offset:x_offset + width] = cropped_image
        
        return padded_image

    def compute_hsv_mean(self, hsv_image):
        non_black_mask = np.all(hsv_image != [0, 0, 0], axis=-1)
        hsv_non_black = hsv_image[non_black_mask]

        if len(hsv_non_black) == 0:
            return 0, 0, 0

        h_mean = np.mean(hsv_non_black[:, 0])
        s_mean = np.mean(hsv_non_black[:, 1])
        v_mean = np.mean(hsv_non_black[:, 2])

        return h_mean, s_mean, v_mean

    def extract_features(self, image):
        # feature engineering
        segmented = self.segment_object(image)

        if segmented is not None and np.any(segmented):
            resized = cv2.resize(segmented, self.resize_dim)
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

            h_mean_all, s_mean_all, v_mean_all = self.compute_hsv_mean(hsv)

            center_region = hsv[14:42, 14:42]
            h_mean_center, s_mean_center, v_mean_center = self.compute_hsv_mean(center_region)

            return np.array([h_mean_all, s_mean_all, v_mean_all, h_mean_center, s_mean_center, v_mean_center])
        else:
            raise ValueError("No object detected in the image")

    def load_model_and_scaler(self, model_filename, scaler_filename):
        model = joblib.load(model_filename)
        scaler = joblib.load(scaler_filename)
        return model, scaler

class ModelPredictor:
    def __init__(self, model, scaler, class_names):
        self.model = model
        self.scaler = scaler
        self.class_names = class_names
    
    def predict(self, features):
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)
        print(self.class_names[prediction[0]])
        return prediction[0]

class NonLinearDeblur:
    def __init__(self, padding=50, alpha=0.8, R=1, snr_h=1, snr_s=0.95, snr_v=0.15):
        self.padding = padding
        self.alpha = alpha
        self.R = R
        self.snr_h = snr_h
        self.snr_s = snr_s
        self.snr_v = snr_v
    
    def calcPSF(self, filterSize):
        # Calculate Point Spread Function
        h = np.zeros(filterSize, np.float32)
        center = (filterSize[0] // 2, filterSize[1] // 2)
        cv2.circle(h, center, self.R, 1.0, -1, 8)
        summa = np.sum(h)
        return h / summa

    def calcWnrFilter(self, h_PSF, nsr):
        # Create Wiener filter
        h_PSF_shifted = np.fft.fftshift(h_PSF)
        H = np.fft.fft2(h_PSF_shifted)
        denom = np.abs(H)**2 + nsr
        WnrFilter = 1 / denom
        return WnrFilter

    def deblur_image(self, inputImg, WnrFilter):
        # Deblur the image
        dft = np.fft.fft2(inputImg)
        dft_shifted = np.fft.fftshift(dft)
        
        WnrFilter_resized = np.transpose(WnrFilter)  # Transpose WnrFilter to match dft_shifted's shape
        complexIH = dft_shifted * WnrFilter_resized
        idft = np.fft.ifft2(np.fft.ifftshift(complexIH))
        return np.abs(idft)

    def remove_salt_and_pepper_noise(self, image, kernel_size=1):
        # Remove salt and pepper noise (Median filtering)
        return cv2.medianBlur(image, kernel_size)

    def deblur_and_enhance(self, input_image):
        # Image deblurring and enhancement
        image_hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
        filterSize = (image_hsv.shape[1], image_hsv.shape[0])
        
        PSF = self.calcPSF(filterSize)
        WnrFilter_h = self.calcWnrFilter(PSF, self.snr_h)
        WnrFilter_s = self.calcWnrFilter(PSF, self.snr_s)
        WnrFilter_v = self.calcWnrFilter(PSF, self.snr_v)

        deblurred_image_hsv = image_hsv.copy()
        deblurred_image_hsv[:, :, 0] = self.deblur_image(image_hsv[:, :, 0], WnrFilter_h)  # H channel
        deblurred_image_hsv[:, :, 1] = self.deblur_image(image_hsv[:, :, 1], WnrFilter_s)  # S channel
        deblurred_image_hsv[:, :, 2] = self.deblur_image(image_hsv[:, :, 2], WnrFilter_v)  # V channel

        deblurred_image = cv2.cvtColor(deblurred_image_hsv, cv2.COLOR_HSV2BGR)
        deblurred_image_no_noise = self.remove_salt_and_pepper_noise(deblurred_image)
        lighted_image = cv2.convertScaleAbs(deblurred_image_no_noise, alpha=1.2, beta=10)

        return lighted_image

class ObjectSegmentation:
    def __init__(self, padding=50, alpha=0.8):
        self.padding = padding
        self.alpha = alpha

    def segment_object(self, frame, bg):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]
        _, otsu_thresholded = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((15, 15), np.uint8)
        closed_mask = cv2.morphologyEx(otsu_thresholded, cv2.MORPH_CLOSE, kernel)
        kernel = np.ones((27, 27), np.uint8)
        opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((9, 9), np.uint8)
        inflated_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_DILATE, kernel)
        contours, _ = cv2.findContours(inflated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        convex_hull_mask = np.zeros_like(inflated_mask)
        largest_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest_contour)
        cv2.drawContours(convex_hull_mask, [hull], -1, (255), thickness=cv2.FILLED)
        
        frame_with_hull = cv2.bitwise_and(frame, frame, mask=convex_hull_mask)
        black_background = np.zeros_like(frame)
        final_image = cv2.add(black_background, frame_with_hull)

        x, y, w, h = cv2.boundingRect(hull)
        x_min = max(x - self.padding, 0)
        y_min = max(y - self.padding, 0)
        x_max = min(x + w + self.padding, frame.shape[1])
        y_max = min(y + h + self.padding, frame.shape[0])
        cropped_image = final_image[y_min:y_max, x_min:x_max]
        
        height, width = cropped_image.shape[:2]
        target_height = height
        target_width = int(target_height * 4 / 3)

        if target_width < width:
            target_width = width
            target_height = int(target_width * 3 / 4)

        top_padding = (target_height - height) // 2
        left_padding = (target_width - width) // 2

        padded_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        padded_image[top_padding:top_padding + height, left_padding:left_padding + width] = cropped_image
        padded_image = cv2.resize(padded_image, (384, 288))

        mask = cv2.cvtColor(padded_image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        
        masked_bg = cv2.bitwise_and(bg, bg, mask=mask_inv)
        masked_padded_image = cv2.bitwise_and(padded_image, padded_image, mask=mask)
        final_image = cv2.add(masked_padded_image, masked_bg)
        
        blurred_final_image = cv2.GaussianBlur(final_image, (3, 3), 0)
        bg_float = bg.astype(np.float32) / 255.0
        fg_float = blurred_final_image.astype(np.float32) / 255.0
        blended_image = (bg_float * (1 - self.alpha)) + (fg_float * self.alpha)
        blended_image = (blended_image * 255).astype(np.uint8)
        
        oil_painting_image = cv2.xphoto.oilPainting(blended_image, 4, 1)
        
        return oil_painting_image


def main():
    cap = cv2.VideoCapture(0)
    
    model_filename = 'ada_boost_model.joblib'
    scaler_filename = 'scaler.joblib'
    class_names = ['', 'pikachu', 'squirtle', 'charmander', 'bulbasaur', 'jiggypuff', 'psyduck']
    
    bg = cv2.imread('bg.jpg')
    bg = cv2.resize(bg, (384, 288))
    
    deblurrer = NonLinearDeblur()
    segmenter = ObjectSegmentation()
    image_processor = ImageProcessor(resize_dim=(56, 56))
    model, scaler = image_processor.load_model_and_scaler(model_filename, scaler_filename)
    model_predictor = ModelPredictor(model, scaler, class_names)

    try:
        while True:
            ret, image = cap.read()
            if not ret:
                break
            
            # extract features and predict
            features = image_processor.extract_features(image)
            prediction = model_predictor.predict(features)
            
            # beautify image
            deblurred_image = deblurrer.deblur_and_enhance(image)
            final_result = segmenter.segment_object(deblurred_image, bg)
            
            cv2.imshow("Final Result", final_result)
            cv2.waitKey(10)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
