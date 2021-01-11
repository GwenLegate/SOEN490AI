import cv2
import numpy as np
import base64

def data_uri_to_cv_image(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return img

# Offset will be the value of the area we are going to crop. To be sent by the backend or we could hardcode it here
def crop_picture(img, offset):
    img_h, img_w, _ = img.shape
    crop = img[0:offset, offset:img_w]

    return crop

def grayscale(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray_img

# (5,5): Height and width of the kernel used to convolve the image to blur it. It can be tweaked for better results
# The GaussianBlur accepts the standard deviation of X and Y. When 0 is provided, they would be calculated from the kernel size
def apply_gaussian_blur(img):
    gaussian = cv2.GaussianBlur(img, (5,5), 0)

    return gaussian

# Opening is the process of applying erosion followed by dilation to remove noise
# Erosion is the process of thinning the image by removing pixels from the image
# Dilation is the process of dilating the image by adding pixels to it
def opening_img(img):
    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    return opening

# kernel_sharpening: Arbitrary matrix. Should be normalized (i.e matrix should sum up to 1 to keep brightness of the image intact)
def sharpen(img):
    kernel_sharpening = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(img,-1 , kernel_sharpening)

    return sharpened

# ================ MAIN ===================
def process_image(img, offset):
    cropped_img = crop_picture(data_uri_to_cv_image(img), offset)
    no_noise = opening_img(grayscale(apply_gaussian_blur(cropped_img)))
    sharpened = sharpen(no_noise)

    return sharpened