import torch
import sklearn.metrics as metrics
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import utilities.data_processing as data
import constants as c
from alphabet_model.train_alphabet_model import Net

'''def test(size):
    random_start = np.random.randint(len(alphabet_test_data) - size)
    X, y = test_features[random_start:random_start + size], test_labels[random_start:random_start + size]
    with torch.no_grad():
        test_accuracy, test_loss = feed_model(X.view(-1, 1, 28, 28).to(device), y.to(device))
    return test_accuracy, test_loss'''