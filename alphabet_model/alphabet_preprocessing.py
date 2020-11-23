import sys, os
from sklearn.preprocessing import StandardScaler
import numpy as np
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
            alpha_X = StandardScaler().fit_transform(alpha_X.reshape(-1, 200 * 200))  # standardize data around mean = 0
            np.save(fname, alpha_X.reshape(-1, 200, 200))
        print(alpha_X.shape)

'''creates and saves label array in one hot vector format.  All letters have 3000 instances except J and Z which have 0.'''
def create_alphabet_labels():
    alpha_y = np.zeros(3000, dtype=int)
    for i in range(1, 25):
        if not i == 9:
            arr = np.full(3000, i)
            alpha_y = np.concatenate((alpha_y, arr))
    alpha_y = data.one_hot_vector(alpha_y, num_classes=26)
    np.save('training_labels.npy', alpha_y)
