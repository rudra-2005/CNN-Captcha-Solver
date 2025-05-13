import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import string

# Load model
model = load_model("captcha_solver.h5")

# Class labels
info = list(string.digits + string.ascii_uppercase + string.ascii_lowercase)

# Preprocessing functions
def t_img(img):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 145, 0)

def c_img(img):
    kernel = np.ones((5, 2), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def b_img(img):
    return cv2.GaussianBlur(img, (1, 1), 0)

def predict_captcha(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image not found.")
        return


    img = t_img(img)
    img = c_img(img)
    img = b_img(img)



    image_list = [
        img[10:50, 30:50],
        img[10:50, 50:70],
        img[10:50, 70:90],
        img[10:50, 90:110],
        img[10:50, 110:130]
    ]

    prediction = ''
    for char_img in image_list:
        resized = cv2.resize(char_img, (20, 40)) 
        resized = img_to_array(resized)
        if resized.shape[-1] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            resized = resized.reshape(40, 20, 1)
        elif len(resized.shape) == 2:
            resized = resized.reshape(40, 20, 1)

        resized = resized.astype("float32") / 255.0
        resized = np.expand_dims(resized, axis=0) 

        y_pred = model.predict(resized)
        predicted_index = np.argmax(y_pred)
        prediction += info[predicted_index]

    print("CAPTCHA Prediction:", prediction)
    print("Filename:", img_path[-9:])

if __name__ == "__main__":
    img_path = input("Enter the path to the CAPTCHA image: ").strip()
    predict_captcha(img_path)
