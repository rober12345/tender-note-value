import cv2
import numpy as np
from PIL import Image

def load_and_preprocess(image_path, img_size=(128, 128)):
    image = Image.open(image_path).convert('L').resize(img_size)
    image = np.array(image) / 255.0
    image = image.reshape(*img_size, 1)
    return image
