import time
from random import seed
from random import randint
import cv2
from skimage.util import random_noise
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import constants as c

GAUSSIAN = "gaussian"
SALT_PEPPER = "s&p"
POISSON = "poisson"

def generate_random_noise(img, mode, seed):
    return random_noise(img, mode=mode, seed=seed)

def check_percentage(percentage):
    random_percentage = randint(1, 100)
    return percentage > random_percentage

# ==================================== MAIN ====================================
'''
    Applies gaussian, salt and pepper or poisson noise randomly to a given picture. Will randomly apply one of the noises randomly using the time as seed.
    val_seed is the random seed to have randomness to the noise application.

    Return noisy image as float values (Float values from 0 to 1)
'''
def apply_noise(img_path):
    seed(time.time()) # For generating random numbers based on system time

    img = cv2.imread(img_path, 0)

    # Do not apply the noise filter in 90% of images
    if check_percentage(10):
        return img
    else:
        '''
            0: gaussian noise
            1: salt and pepper noise
            2: poisson
        '''
        val = randint(0, 2)
        val_seed = randint(0, 100) # For pseudo randomness in the noise application
        if val == 0:
            noisy_image = generate_random_noise(img, GAUSSIAN, val_seed)
        elif val == 1:
            noisy_image = generate_random_noise(img, SALT_PEPPER, val_seed)
        else:
            noisy_image = generate_random_noise(img, POISSON, val_seed)

        return noisy_image

'''img = apply_noise(c.GWEN_B)
plt.imshow(img)
plt.show()'''