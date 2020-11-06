import torch
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import pandas as pd
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

'''creates plots of loss training and loss testing and accuracy training and accuracy testing over time'''
def plot_accuracy_and_loss(logfile_path):
    data = pd.read_csv(logfile_path).to_numpy()
    times = data[:, -6]
    epoch = data[:, -6:-4]
    train_accuracy = data[:, -4]
    train_loss = data[:, -3]
    test_accuracy = data[:, -2]
    test_loss = data[:, -1]
    fig, axs = plt.subplots(2)

    '''plot epochs as vertical lines'''
    filter_epoch = np.array(epoch[0, :])
    last = epoch[0, 1]
    for e in epoch:
        if last != e[1]:
            np.vstack((filter_epoch, e))
            last = e[1]

    axs[0].plot(times, test_accuracy, label="train accuracy")
    axs[0].plot(times, train_accuracy, label="test accuracy")
    axs[0].set_title("Accuracy")
    axs[0].set(xlabel="time", ylabel="Accuracy")
    axs[0].vlines(filter_epoch[0], 0, 1, linestyles="dotted", label="epochs")
    axs[0].legend()

    axs[1].plot(times, test_loss, label="train loss")
    axs[1].plot(times, train_loss, label="test loss")
    axs[1].set_title("Loss")
    axs[1].set(xlabel="time", ylabel="Loss")
    axs[0].vlines(filter_epoch[0], 0, 1, linestyles="dotted", label="epochs")
    axs[1].legend()

    plt.show()