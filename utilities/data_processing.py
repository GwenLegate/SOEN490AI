# Contains utility functions for data access

# Import statements
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# returns a usable format of the input data.  Input (str): path to fil to be loaded, type (str): type of file input
# currently only accepts *.csv, can be extended to other data formats if required later in development
def get_data(path, type = 'csv'):
    if type == 'csv':
        return pd.read_csv(path).to_numpy()
    else:
        print('type must be \'csv\'')

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
