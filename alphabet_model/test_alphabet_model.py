import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import utilities.data_processing as data
import utilities.evaluation_metrics as eval
import constants as c
from alphabet_model.train_alphabet_model import Net

''' check for GPU, if no GPU, use CPU '''
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

'''global variables and flags to control execution'''
PLOT_ACCURACY_LOSS = False

def predict_test_set():
    '''Import data from source file and separate into features and labels'''
    alphabet_test_data = data.get_data(c.FILE_TEST_ALPHABET)
    alphabet_testX, alphabet_testy = alphabet_test_data[:, 1:], alphabet_test_data[:, 0]

    '''normalize pixel values'''
    alphabet_testX = alphabet_testX / 255.0

    '''convert to tensor to be put through the array.  Need to convert to torch.DoubleTensor to avoid type error when 
    feeding data through the model'''
    test_features = torch.from_numpy(alphabet_testX).view(-1, 28, 28).type('torch.FloatTensor')
    test_labels = torch.from_numpy(alphabet_testy)
    ''' load the trained model'''
    alphabet_cnn = Net()
    alphabet_cnn.load_state_dict(torch.load(c.MODEL_SAVE_PATH + "/alphabet_model.pt", map_location=device))
    return alphabet_cnn(test_features.view(-1, 1, 28, 28)).detach().numpy()

'''generate plot of accuracy and loss over time'''
if PLOT_ACCURACY_LOSS:
    data.plot_accuracy_and_loss(os.getcwd()+"/alphabet_model.log")

