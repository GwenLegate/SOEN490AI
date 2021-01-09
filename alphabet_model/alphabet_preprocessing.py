import sys, os
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import utilities.data_processing as data
import constants as c

''' returns numpy arrays of training and testing features and labels of the MNIST dataset '''
def process_mnist():
    # Import data from MNIST source file and convert to numpy arrays for features and labels
    alphabet_train_data = data.get_data(c.FILE_TRAIN_ALPHABET)
    alphabet_test_data = data.get_data(c.FILE_TEST_ALPHABET)
    alphabet_testX, alphabet_testy = alphabet_test_data[:, 1:], alphabet_test_data[:, 0]
    alphabet_trainX, alphabet_trainy = alphabet_train_data[:, 1:], alphabet_train_data[:, 0]

    # reformat y to one-hot-vector-format for comparison with outputs
    alphabet_testy = data.one_hot_vector(alphabet_testy, num_classes=26)
    alphabet_trainy = data.one_hot_vector(alphabet_trainy, num_classes=26)

    # normalize pixel values of features
    alphabet_train_features = alphabet_trainX / 255.0
    alphabet_test_features = alphabet_testX / 255.0
    return alphabet_trainX, alphabet_trainy, alphabet_testX, alphabet_testy

''' takes an input array "letters", containing the directories to be processed and a string fname to save the output under'''
def preprocess_training_images(letters, fname):
    # preprocess images and add them to the input feature array
    alpha_X = data.get_training_arr(fname)
    print(alpha_X.shape)
    for letter in letters:
        for root, dirs, files in os.walk(c.TRAIN_ALPHABET_IMGS_BASEDIR+letter):
            for name in files:
                print(name)
                px = data.preprocess_image(os.path.join(root, name)).reshape(-1, 200, 200)
                alpha_X = np.append(alpha_X, px, axis=0)
            np.save(fname, alpha_X.reshape(-1, 200, 200))
        print(alpha_X.shape)

''' scale dataset and save'''
def scale(alpha, fname):
    alpha = StandardScaler().fit_transform(alpha.reshape(-1, 200 * 200))  # standardize data around mean = 0
    np.save(fname, alpha)

'''save scalar for future use'''
def save_scalar(fname):
    alpha = data.get_training_arr('alphabet_train_features.npy') # unshuffled, unscaled data
    scalar = StandardScaler().fit(alpha.reshape(-1, 200 * 200)) # fit scaler
    file = open(fname, 'wb')
    pickle.dump(scalar, file)
    file.close()

''' preprocesses images from the training set'''
def preprocess_testing_images():
    alpha_test_X = np.empty((0, 200, 200))
    #num = 0
    for root, dirs, files in os.walk(c.TEST_ALPHABET_IMGS_BASEDIR):
        for name in files:
            print(name)
            px = data.preprocess_image(os.path.join(root, name))
            row, col, = px.shape

            if row == 200 and col == 200:
                px = px.reshape(-1, 200, 200)
                alpha_test_X = np.append(alpha_test_X, px, axis=0)
            else:

                # pad image
                add_r = 200 - row
                add_c = 200 - col
                if not add_r == 0:
                    px = np.vstack((np.zeros((add_r, c)), px))
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
            alpha_y = np.concatenate((alpha_y, arr))
    alpha_y = data.one_hot_vector(alpha_y, num_classes=26)
    np.save('alphabet_training_labels.npy', alpha_y)

'''creates and saves label array in one hot vector format.  All letters have 30 instances except J and Z which have 0.'''
def create_alphabet_test_labels():
    alpha_test_y = np.zeros(30, dtype=int)
    for i in range(1, 25):
        if not i == 9:
            arr = np.full(30, i)
            alpha_test_y = np.concatenate((alpha_test_y, arr))

    alpha_test_y = data.one_hot_vector(alpha_test_y, num_classes=26)
    np.save('alphabet_test_labels.npy', alpha_test_y)

''' shuffles test set before use'''
def shuffle_test_set(test_X, test_y):
    test_X = test_X.reshape(720, -1)
    test_y = data.numeric_class(test_y).reshape(-1, 1)
    xy = np.hstack((test_X, test_y))
    np.random.shuffle(xy)
    test_X, test_y = xy[:, :40000].reshape(-1, 200, 200), xy[:, -1]
    test_y = data.one_hot_vector(test_y.astype(int), num_classes=26)
    return test_X, test_y


def shuffle_train_set(train_X, train_y):
    train_X = train_X.reshape(72000, -1)
    train_y = data.numeric_class(train_y).reshape(-1, 1)
    xy = np.hstack((train_X, train_y))
    np.random.shuffle(xy)
    train_X, train_y = xy[:, :40000].reshape(-1, 200, 200), xy[:, -1]
    train_y = data.one_hot_vector(train_y.astype(int), num_classes=26)
    return train_X, train_y