# Contains utility functions for data access

# Import statements
import pandas as pd

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