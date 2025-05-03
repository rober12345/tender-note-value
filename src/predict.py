import sys
import numpy as np
from tensorflow.keras.models import load_model
from preprocess import load_and_preprocess

value_model = load_model('../models/value_model.h5')
auth_model = load_model('../models/auth_model.h5')

value_classes = ['20', '50', '100', '200', '500', '1000']  # Adjust if needed

def predict(image_path):
    img = load_and_preprocess(image_path).reshape(1, 128, 128, 1)

    value_pred = value_model.predict(img)
    auth_pred = auth_model.predict(img)

    value = value_classes[np.argmax(value_pred)]
    is_fake = auth_pred[0][0] > 0.5

    print(f"Detected Value: ${value} MXN")
    print("⚠️ FAKE!" if is_fake else "✅ Authentic")

if __name__ == '__main__':
    predict(sys.argv[1])  # Example: python predict.py test.jpg
