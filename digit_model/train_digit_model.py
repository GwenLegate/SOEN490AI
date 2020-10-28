import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import utilities.data_processing as data
import constants as c

'''Flags to control execution'''
CHECK_BALANCE = False
TRAIN = False

''' Check for GPU, if no GPU, use CPU'''
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

''' Define hyper parameters'''
KERNEL = 5
LEARNING_RATE = 0.001
EPOCHS = 25
BATCH_SIZE = 72

'''input parameters'''
MODEL_NAME = f"digit_model-{int(time.time())}" # make logfile (time.time() gives a time stamp so you can have a history)

''' Import data from source file and split into training and testing sets'''
digits_X = data.get_data(c.FILE_DIGIT_FEATURES, type='npy')
digits_y = data.get_data(c.FILE_DIGIT_LABELS, type='npy')

