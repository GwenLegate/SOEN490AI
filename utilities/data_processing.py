''' Import statements '''
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from image_processing.angels_image_processing_tool import process_image
import random
import copy

''' checks the dataset is balanced, prints out the input dictionary updated with the total amount of each class
in the training set.
Input:
dict(dictionary): initialized with all of the classes with their total count set to 0. ex. class:0
labels(numpy array):  array containing all the labels of the training set instances'''
def check_balance(dict, labels):
    total = 0
    for y in labels:
        dict[int(y)] += 1
        total += 1
    print(dict)

'''given an image path, the function returns a numpy array of the image with gaussian blur technique, greyscaled, 
centered around the mean pixel value and normalized pixel values between 0-1 
0 and 1'''
def preprocess_image(image):
    # convert image to greyscale and np array
    try:
        img = np.asarray(Image.open(image).convert('L'))
    except AttributeError:
        img = Image.fromarray(image)
        img = np.asarray(img.convert('L'))

    #gaussian blur and sharpen edges then flatten image
    img = process_image(img)
    x, y = img.shape
    img = np.ravel(img)

    # center data around the mean
    avg = np.average(img)
    img = img - avg

    # normalize using MinMax scaling
    max, min = np.amax(img), np.amin(img)
    img = (img - min) / (max - min)

    return img.reshape(x, y)

'''reformat y to one-hot-vector-format'''
def one_hot_vector(y, num_classes):
    reformat_y = np.zeros((len(y), num_classes)).astype(int)
    j = 0
    while j < len(y):
        reformat_y[j][y[j]] = 1
        j += 1
    return reformat_y


''' turns one-hot-vector into numberic label'''
def numeric_class(y):
    arr = np.empty(0)
    for i in y:
        arr = np.append(arr, np.argmax(i)).astype(int)
    return arr

''' check if file exists and returns loaded numpy array if it does, otherwise it returns an empty array'''
def get_training_arr(file):
    try:
        np.load(file, allow_pickle=True)
    except FileNotFoundError:
        return np.empty((0, 200, 200))
    else:
        return np.load(file, allow_pickle=True)

''' shuffles dataset before use.  Classes is the number of outcomes possible in the classification of the set and 
num_samples is the number of images in the set'''

def shuffle_set(X, y, classes, num_samples):
    X = X.reshape(num_samples, -1)
    y = numeric_class(y).reshape(-1, 1)
    xy = np.hstack((X, y))
    np.random.shuffle(xy)
    X, y = xy[:, :40000].reshape(-1, 200, 200), xy[:, -1]
    y = one_hot_vector(y.astype(int), num_classes=classes)
    return X, y

'''Adding noise to the images over such a large dataset caused memory problems (too large for RAM).  This methon does an
inplace swap of two seperate datasets so that samples are adequately shuffled before use.'''
def swap(feat_X1, label_y1, feat_X2, label_y2):
    arr_len, _ = label_y1.shape
    rand_len = int(arr_len / 2)
    rand = random.sample(range(arr_len), rand_len)
    for i in rand:
        lab_temp = copy.deepcopy(label_y1[i, :])
        feat_temp = copy.deepcopy(feat_X1[i, :, :])
        label_y1[i, :] = label_y2[i, :]
        feat_X1[i, :, :] = feat_X2[i, :, :]
        label_y2[i, :] = lab_temp
        feat_X2[i, :, :] = feat_temp
    return feat_X1, label_y1, feat_X2, label_y2



