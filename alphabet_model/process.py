import sys, os
from sklearn.preprocessing import StandardScaler
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import utilities.data_processing as data
import constants as c
process = ["K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]

for p in process:
    # preprocess images and add them to the input feature array
    for root, dirs, files in os.walk(c.TRAIN_ALPHABET_IMGS_BASEDIR+p):
        for name in files:
            print(name)
            px = data.preprocess_image(os.path.join(root, name)).reshape(-1, 200, 200)
            alpha_X = np.append(alpha_X, px, axis=0)
        alpha_X = StandardScaler().fit_transform(alpha_X.reshape(-1, 200 * 200))  # standardize data around mean = 0
        np.save('training_data.npy', alpha_X.reshape(-1, 200, 200))

    print(alpha_X.shape)
