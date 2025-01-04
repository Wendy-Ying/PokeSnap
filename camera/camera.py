import picamera2
from picamera2 import Picamera2
import time
import RPi.GPIO as GPIO

# Define GPIO pins
BUTTON_CAPTURE_PIN = 5  # GPIO5
BUTTON_STOP_PIN = 6     # GPIO6

def setup_gpio():
    GPIO.setmode(GPIO.BCM)  # Use BCM pin numbering
    GPIO.setup(BUTTON_CAPTURE_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(BUTTON_STOP_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def capture_photo(picam2, output_file):
    # Configure the camera
    config = picam2.create_still_configuration()
    picam2.configure(config)

    # Start the camera
    picam2.start()
    #print("Camera warming up...")
    #time.sleep(2)  # Allow the camera to warm up

    # Capture the photo
    print(f"Capturing photo and saving to {output_file}...")
    picam2.capture_file(output_file)

    # Stop the camera
    picam2.stop()
    print("Photo captured successfully!")

def main():
    setup_gpio()
    picam2 = Picamera2()
    photo_count = 0  # Counter for unique file names

    try:
        while True:
            print("Press the bottom to take a photo")
            # Check the stop button
            if GPIO.input(BUTTON_STOP_PIN) == GPIO.LOW:
                print("Stop button pressed. Exiting program.")
                break

            # Check the capture button
            if GPIO.input(BUTTON_CAPTURE_PIN) == GPIO.LOW:
                photo_count += 1
                output_filename = f"captured_image_{photo_count}.jpg"
                capture_photo(picam2, output_filename)

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
