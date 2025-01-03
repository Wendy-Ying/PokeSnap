import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
import numpy as np
from vit_model import vit_base_patch16_224_in21k as create_model

# Initialize MediaPipe Hand Detection
mphands = mp.solutions.hands
hands = mphands.Hands()

def segment_object(frame):
    """
    Segments the hand and face region from the input image using color thresholds.
    """
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

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the transformation to match the model's input size
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # Image path
    img_path = "enhanced_chinese_0526/G/hsv21.jpg"
    assert os.path.exists(img_path), f"File '{img_path}' does not exist."
    img = Image.open(img_path)

    # Display the original image
    plt.imshow(img)
    plt.title("Original Image")
    plt.show()

    # Segment the hands and face using the defined segmentation function
    image = np.array(img)
    image = segment_object(image)

    # Detect hand landmarks using MediaPipe
    h, w, _ = image.shape
    framergbanalysis = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resultanalysis = hands.process(framergbanalysis)
    if resultanalysis.multi_hand_landmarks:
        for handLMsanalysis in resultanalysis.multi_hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            x_vals = []
            y_vals = []

            for lmanalysis in handLMsanalysis.landmark:
                x, y = int(lmanalysis.x * w), int(lmanalysis.y * h)
                x_vals.append(x)
                y_vals.append(y)

            x_min, x_max = min(x_vals) - 20, max(x_vals) + 20
            y_min, y_max = min(y_vals) - 20, max(y_vals) + 20

            x_min = max(x_min, 0)
            x_max = min(x_max, w)
            y_min = max(y_min, 0)
            y_max = min(y_max, h)

        if x_min < x_max and y_min < y_max:
            # Crop the image around the detected hand region
            image = image[y_min:y_max, x_min:x_max]
            img = cv2.resize(image, (224, 224))

    # Convert to PIL Image to apply the same transformation as training
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Apply the transformations
    img = data_transform(img)

    # Expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # Load class indices from JSON file
    json_path = 'class_indices.json'
    assert os.path.exists(json_path), f"File '{json_path}' does not exist."
    
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # Create the model
    model = create_model(num_classes=30, has_logits=False).to(device)

    # Load model weights
    model_weight_path = "weights/model-4.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # Inference phase: Disable gradient computation
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    # Print the prediction results
    print_res = f"Predicted Class: {class_indict[str(predict_cla)]}   Probability: {predict[predict_cla].numpy():.3f}"
    plt.title(print_res)

    # Print all class predictions
    for i in range(len(predict)):
        print(f"Class: {class_indict[str(i)]:10}   Probability: {predict[i].numpy():.3f}")

    # Display the result
    plt.show()

if __name__ == '__main__':
    main()
