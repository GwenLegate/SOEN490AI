import sys, os
import numpy as np
import pickle
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from utilities.data_processing import get_training_arr, preprocess_image, get_data, one_hot_vector, numeric_class
import constants as c

''' takes an input array "letters", containing the directories to be processed and a string fname to save the output under'''
def preprocess_training_images(letters, fname):
    # preprocess images and add them to the input feature array
    alpha_X = get_training_arr(fname)
    print(alpha_X.shape)
    for letter in letters:
        for root, dirs, files in os.walk(c.TRAIN_ALPHABET_IMGS_BASEDIR+letter):
            for name in files:
                print(name)
                px = preprocess_image(os.path.join(root, name))
                row, col, = px.shape

                if row == 200 and col == 200:
                    px = px.reshape(-1, 200, 200)
                    alpha_X = np.append(alpha_X, px, axis=0)
                else:

                    # pad image
                    add_r = 200 - row
                    add_c = 200 - col
                    if not add_r == 0:
                        px = np.vstack((np.zeros((add_r, col)), px))
                    if not add_c == 0:
                        px = np.hstack((np.zeros((200, add_c)), px))
                    px = px.reshape(-1, 200, 200)
                    alpha_X = np.append(alpha_X, px, axis=0)

            np.save(fname, alpha_X.reshape(-1, 200, 200))
        print(alpha_X.shape)

''' preprocesses images from the training set'''
def preprocess_testing_images():
    alpha_test_X = np.empty((0, 200, 200))
    #num = 0
    for root, dirs, files in os.walk(c.TEST_ALPHABET_IMGS_BASEDIR):
        for name in files:
            print(name)
            px = preprocess_image(os.path.join(root, name))
            row, col, = px.shape

            if row == 200 and col == 200:
                px = px.reshape(-1, 200, 200)
                alpha_test_X = np.append(alpha_test_X, px, axis=0)
            else:

                # pad image
                add_r = 200 - row
                add_c = 200 - col
                if not add_r == 0:
                    px = np.vstack((np.zeros((add_r, col)), px))
                if not add_c == 0:
                    px = np.hstack((np.zeros((200, add_c)), px))
                px = px.reshape(-1, 200, 200)
                alpha_test_X = np.append(alpha_test_X, px, axis=0)
    np.save('alpha_test_inputs.npy', alpha_test_X)

'''creates and saves label array in one hot vector format.  All letters have 3000 instances except J and Z which have 0.'''
def create_alphabet_train_labels():
    alpha_y = np.zeros(3000, dtype=int)
    for i in range(1, 25):
        if not i == 9:
            arr = np.full(3000, i)
            if i == 20:
                arr = np.full(2114, i)
            alpha_y = np.concatenate((alpha_y, arr))

    alpha_y = one_hot_vector(alpha_y, num_classes=26)
    np.save('alpha_train_labels.npy', alpha_y)

def create_one_vs_rest_train_labels():
    one = np.zeros(3000, dtype=int)
    rest = np.ones(3000, dtype=int)
    alpha_y = np.concatenate((one, rest))

    alpha_y = one_hot_vector(alpha_y, num_classes=2)
    np.save('one_vs_rest_train_labels.npy', alpha_y)

'''creates and saves label array in one hot vector format.  All letters have 30 instances except J and Z which have 0.'''
def create_alphabet_test_labels():
    alpha_test_y = np.zeros(30, dtype=int)
    for i in range(1, 25):
        if not i == 9:
            arr = np.full(30, i)
            alpha_test_y = np.concatenate((alpha_test_y, arr))

    alpha_test_y = one_hot_vector(alpha_test_y, num_classes=26)
    np.save('alpha_test_labels.npy', alpha_test_y)

''' shuffles test set before use'''
def shuffle_test_set(test_X, test_y, classes):
    test_X = test_X.reshape(720, -1)
    test_y = numeric_class(test_y).reshape(-1, 1)
    xy = np.hstack((test_X, test_y))
    np.random.shuffle(xy)
    test_X, test_y = xy[:, :40000].reshape(-1, 200, 200), xy[:, -1]
    test_y = one_hot_vector(test_y.astype(int), num_classes=classes)
    return test_X, test_y


def shuffle_train_set(train_X, train_y, classes, num_samples):
    train_X = train_X.reshape(num_samples, -1)
    train_y = numeric_class(train_y).reshape(-1, 1)
    xy = np.hstack((train_X, train_y))
    np.random.shuffle(xy)
    train_X, train_y = xy[:, :40000].reshape(-1, 200, 200), xy[:, -1]
    train_y = one_hot_vector(train_y.astype(int), num_classes=classes)
    return train_X, train_y
