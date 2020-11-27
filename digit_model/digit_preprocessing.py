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
                px = data.preprocess_image(os.path.join(root, name))
                row, col, = px.shape

                if row == 100 and col == 100:
                    px = px.reshape(-1, 100, 100)
                    digits_X = np.append(digits_X, px, axis=0)
                else:
                    # pad image
                    add_r = 100 - row
                    add_c = 100 - col
                    if add_c >=0 and add_r >=0:
                        if not add_r == 0:
                            px = np.vstack((np.zeros((add_r, c)), px))
                        if not add_c == 0:
                            px = np.hstack((np.zeros((100, add_c)), px))
                        px = px.reshape(-1, 100, 100)
                        digits_X = np.append(digits_X, px, axis=0)
                    elif add_c > -51 and add_r > -51:
                        # scale image
                        if not add_r == 0:
                            add_r *= -1
                            r_top = int(add_r/2)
                            r_bottom = add_r - r_top
                            px = px[r_top: -r_bottom, :]
                        if not add_c == 0:
                            add_c *= -1
                            c_top = int(add_c / 2)
                            c_bottom = add_c - c_top
                            px = px[c_top: -c_bottom, :]
                        px = px.reshape(-1, 100, 100)
                        digits_X = np.append(digits_X, px, axis=0)
                    else:
                        with open("unprocessed_digit_imgs.log", "a+") as f:
                            f.write("%s, %s\n" % (digit, name))


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
        if i == 7:
            # 3 images in 7, not added
            arr = np.full(get_digit_amount(str(i)), i)
            arr = arr[3:]
            digits_y = np.concatenate((digits_y, arr))
        else:
            arr = np.full(get_digit_amount(str(i)), i)
            digits_y = np.concatenate((digits_y, arr))

    digits_y = data.one_hot_vector(digits_y, num_classes=10)
    np.save('digit_labels.npy', digits_y)

''' shuffles and splits set before use'''
def shuffle_and_split_set(X, y, no_inputs, test_set_size):
    X = X.reshape(no_inputs, -1)
    y = data.numeric_class(y).reshape(-1, 1)
    xy = np.hstack((X, y))
    np.random.shuffle(xy)
    X, y = xy[:, :10000].reshape(-1, 100, 100), xy[:, -1]
    y = data.one_hot_vector(y.astype(int), num_classes=10)
    num_test = int(test_set_size*no_inputs)

    np.save('digit_train_labels.npy', y[num_test:, :])
    np.save('digit_test_labels.npy', y[:num_test, :])
    np.save('digit_train_features.npy', X[num_test:, :])
    np.save('digit_test_features.npy', X[:num_test, :])

'''scales and saves input array'''
def scale(feature_arr, fname):
    digits_X = StandardScaler().fit_transform(feature_arr.reshape(-1, 100 * 100))  # standardize data around mean = 0
    np.save(fname, digits_X)
