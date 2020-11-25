import sys, os
from sklearn.preprocessing import StandardScaler
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import utilities.data_processing as data
import constants as c

''' takes an input array "digits", containing the directories to be processed and a string fname to save the output under'''
def preprocess_training_images(digits, fname):
    # preprocess images and add them to the input feature array
    digits_X = data.get_training_arr(fname)
    print(digits_X.shape)
    for digit in digits:
        for root, dirs, files in os.walk(c.TRAIN_DIGIT_IMGS_BASEDIR+digit):
            for name in files:
                print(name)
                px = data.preprocess_image(os.path.join(root, name)).reshape(-1, 100, 100)
                digits_X = np.append(digits_X, px, axis=0)
            digits_X = StandardScaler().fit_transform(digits_X.reshape(-1, 100 * 100))  # standardize data around mean = 0
            np.save(fname, digits_X.reshape(-1, 100, 100))
        print(digits_X.shape)

''' determine number of images for each digit, takes digit as a string'''
def get_digit_amount(digit):
    amt = 0
    for root, dirs, files in os.walk(c.TRAIN_DIGIT_IMGS_BASEDIR + digit):
        for _ in files:
            amt += 1
    return amt

'''creates and saves label array in one hot vector format.'''
def create_digit_labels():
    digits_y = np.zeros(get_digit_amount("0"), dtype=int)
    for i in range(1, 10):
        arr = np.full(get_digit_amount(str(i)), i)
        digits_y = np.concatenate((digits_y, arr))
    digits_y = data.one_hot_vector(digits_y, num_classes=10)
    np.save('digit_training_labels.npy', digits_y)
