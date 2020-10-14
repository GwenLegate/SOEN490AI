# This script tests the performance of the trained model and creates statistical plots for visualization

# Import statements
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import utilities.data_processing as data
import constants as c

# Import data from source file and seperate into features and labels
alphabet_test_data = data.get_data(c.FILE_TEST_ALPHABET)
alphabet_testX, alphabet_testy = alphabet_test_data[:, 1:], alphabet_test_data[:, 0]
