import torch
import PIL
import numpy as np
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

'''real image test'''

alphabet_cnn = Net()
alphabet_cnn.load_state_dict(torch.load(c.MODEL_SAVE_PATH + "/alphabet_model.pt", map_location=device))

alpha_key = {0:"A", 1:"B", 2:"C", 3:"D", 4:"E", 5:"F", 6:"G", 7:"H", 8:"I", 9:"J", 10:"K", 11:"L", 12:"M", 13:"N", 14:"O", 15:"P", 16:"Q", 17:"R", 18:"S", 19:"T", 20:"U", 21:"V", 22:"W", 23:"X", 24:"Y", 25:"Z"}

img = PIL.Image.open(c.GWEN_W).convert("L")
web_l = np.array(img) / 255.0
w, l = web_l.shape

def predict_az(input):
    input_tensor = torch.from_numpy(input).view(-1, w, l).type('torch.FloatTensor')
    predict_vect = alphabet_cnn(input_tensor.view(-1, 1, w, l)).detach().numpy()
    predict_val = np.argmax(predict_vect)
    return alpha_key[predict_val]

print(predict_az(web_l))


