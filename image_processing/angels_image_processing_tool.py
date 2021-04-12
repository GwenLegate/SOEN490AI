import cv2
import numpy as np


def grayscale(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray_img

def apply_gaussian_blur(img):
    gaussian = cv2.GaussianBlur(np.float32(img), (5,5), 0)

    return gaussian

def opening_img(img):
    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    return opening

def sharpen(img):
    kernel_sharpening = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(img,-1 , kernel_sharpening)

    return sharpened

# ================ MAIN ===================
def process_image(img):
    no_noise = opening_img(apply_gaussian_blur(img))
    sharpened = sharpen(no_noise)

    return sharpened