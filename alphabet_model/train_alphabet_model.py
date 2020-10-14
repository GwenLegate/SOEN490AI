# This script contains the training parameters and algorithms used to train the ASL alphabet model

# Import statements
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import utilities.data_processing as data
import constants as c

# flags
CHECK_BALANCE = False
TRAIN = True

# Import data from source file
alphabet_train_data = data.get_data(c.FILE_TRAIN_ALPHABET)
alphabet_trainX, alphabet_trainy = alphabet_train_data[:, 1:], alphabet_train_data[:, 0]

# Check balance of dataset
sign_totals = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0, 21:0, 22:0, 23:0, 24:0, 25:0}

if(CHECK_BALANCE):
   data.check_balance(sign_totals, alphabet_trainy)
