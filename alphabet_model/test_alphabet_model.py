import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import utilities.data_processing as data
import constants as c
import alphabet_model.train_alphabet_model

''' check for GPU, if no GPU, use CPU '''
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


'''Import data from source file and separate into features and labels'''
alphabet_test_data = data.get_data(c.FILE_TEST_ALPHABET)
alphabet_testX, alphabet_testy = alphabet_test_data[:, 1:], alphabet_test_data[:, 0]

''' load the trained model'''
alphabet_cnn = alphabet_model.Net()
alphabet_cnn.load_state_dict(torch.load(c.MODEL_SAVE_PATH+"/alphabet_model.pt", map_location=device))
alphabet_cnn.eval() # evaluation mode