from picamera2 import Picamera2
import time
import RPi.GPIO as GPIO
from escpos.printer import File
from PIL import Image
from vision import ImageProcessor, ModelPredictor, ObjectSegmentation, NonLinearDeblur
import cv2
import numpy as np
import joblib

from luma.core.interface.serial import i2c, spi
from luma.core.render import canvas
from luma.oled.device import ssd1306, ssd1325, ssd1331, sh1106

# Define GPIO pins
BUTTON_CAPTURE_PIN = 5  # GPIO5
BUTTON_STOP_PIN = 6     # GPIO6

def setup_gpio():
    GPIO.setmode(GPIO.BCM)  # Use BCM pin numbering
    #GPIO.setmode(GPIO.BOARD)
    GPIO.setup(BUTTON_CAPTURE_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(BUTTON_STOP_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def capture_and_print_photo(picam2, printer, output_file):
    # Configure the camera
    config = picam2.create_still_configuration()
    picam2.configure(config)

    # Start the camera
    picam2.start()

    # Capture the photo
    print(f"Capturing photo and saving to {output_file}...")
    picam2.capture_file(output_file)

    # Stop the camera
    picam2.stop()
    print("Photo captured successfully!")

    # Load and print the photo
    try:
        img = Image.open(output_file)
        img = img.resize((384, img.height * 384 // img.width), Image.LANCZOS)
        printer.image(img)
        printer.cut()
        print("Photo printed successfully!")
    except Exception as e:
        print(f"Error printing photo: {e}")

def main():
    setup_gpio()
    picam2 = Picamera2()
    printer = File("/dev/usb/lp0")
    photo_count = 0  # Counter for unique file names

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
    
    serial = i2c(port=1, address=0x3C)
    device = ssd1306(serial)


    try:
        while True:
            print("Press the button to take a photo and print it.")
            # Check the stop button
            if GPIO.input(BUTTON_STOP_PIN) == GPIO.LOW:
                print("Stop button pressed. Exiting program.")
                break

            # Check the capture button
            if GPIO.input(BUTTON_CAPTURE_PIN) == GPIO.LOW:
                photo_count += 1
                output_filename = f"captured_image_{photo_count}.jpg"
                capture_and_print_photo(picam2, printer, output_filename)
                
                # load image
                image = cv2.imread(output_filename)

                # extract features and predict
                features = image_processor.extract_features(image)
                prediction = model_predictor.predict(features)
                print(prediction)
                with canvas(device) as draw:
                    draw.rectangle(device.bounding_box, outline="white", fill="black")
                    draw.text((30, 20), str(prediction), fill="white")
                
                # beautify image
                deblurred_image = deblurrer.deblur_and_enhance(image)
                final_result = segmenter.segment_object(deblurred_image, bg)
                cv2.imshow("Final Result", final_result)
                cv2.waitKey(100)

                # Wait for the button to be released to prevent multiple triggers
                while GPIO.input(BUTTON_CAPTURE_PIN) == GPIO.LOW:
                    time.sleep(0.1)

            # Brief sleep to reduce CPU usage
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting program.")

    finally:
        GPIO.cleanup()
        print("GPIO cleanup complete. Program terminated.")

if __name__ == "__main__":
    main()