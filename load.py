import pandas as pd
from PIL import Image
import os

class ImageData:
    def __init__(self, image, labels):
        self.image = image
        self.labels = labels

    def __repr__(self):
        return f"ImageData(Labels: {self.labels})"

def load():
    # labels in image_data_list[i].labels
    # images in image_data_list[i].image
    
    csv_file_path = 'dataset/train/_classes.csv'
    image_dir_path = 'dataset/train/'
    
    df = pd.read_csv(csv_file_path)
    image_data_list = []

    for index, row in df.iterrows():
        filename = row['filename']
        image_path = os.path.join(image_dir_path, filename)
        if os.path.exists(image_path):
            try:
                image = Image.open(image_path)
                labels = [class_name for class_name, label in row[df.columns[1:]].items() if label == 1]
                image_data = ImageData(image, labels)
                image_data_list.append(image_data)
            except IOError:
                print(f"Error loading image: {filename}")
        else:
            print(f"File not found: {filename}")

    return image_data_list
    
