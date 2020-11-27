import torch
import sklearn
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import utilities.data_processing as data
import utilities.evaluation_metrics as eval
import constants as c
from alphabet_model.train_alphabet_model import Net
from alphabet_model.alphabet_preprocessing import shuffle_test_set

''' check for GPU, if no GPU, use CPU '''
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

''' predict mnist test set'''
def predict_test_set_MNIST():
    '''Import data from source file and separate into features and labels'''
    alphabet_test_data = data.get_data(c.FILE_TEST_ALPHABET, type='csv')
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

'''global variables and flags to control execution'''
PLOT_ACCURACY_LOSS = False
REPROCESS_TEST_IMAGES = False
RECALCULATE_TEST_PREDICTIONS = False


'''testing'''
alphabet_cnn = Net()
alphabet_cnn.load_state_dict(torch.load(c.MODEL_SAVE_PATH + "/alphabet_model.pt", map_location=device))

alpha_key = {0:"A", 1:"B", 2:"C", 3:"D", 4:"E", 5:"F", 6:"G", 7:"H", 8:"I", 9:"J", 10:"K", 11:"L", 12:"M", 13:"N",
             14:"O", 15:"P", 16:"Q", 17:"R", 18:"S", 19:"T", 20:"U", 21:"V", 22:"W", 23:"X", 24:"Y", 25:"Z"}

''' returns predictions from the model.  The default type 1 will return the predicted letter, type 2 will return a 
one-hot-vector prediction for the inputs that can be used for comparison with the labels'''

def predict_az(input, type=1):
    try:
        w, l = input.shape
    except ValueError:
        _, w, l = input.shape

    input_tensor = torch.from_numpy(input).view(-1, w, l).type('torch.FloatTensor')
    predict_vect = alphabet_cnn(input_tensor.view(-1, 1, w, l)).detach().numpy()
    if type == 1:
        predict_val = np.argmax(predict_vect)
        return alpha_key[predict_val]
    if type == 2:
        return predict_vect

''' load testing X and y'''
alpha_test_X = data.get_data('test_inputs.npy')
alpha_test_y = data.get_data('test_labels.npy')

if RECALCULATE_TEST_PREDICTIONS:
    alpha_test_X, alpha_test_y = shuffle_test_set(alpha_test_X, alpha_test_y)
    np.save('shuffled_test_labels.npy', alpha_test_y) # save shuffled order so the saved y corresponds to the predicted y
    alpha_predict_y = predict_az(alpha_test_X, type=2)
    np.save('shuffled_test_predictions.npy', alpha_predict_y)

''' load test predictions'''
alpha_predict_y = data.get_data("shuffled_test_predictions.npy")

y1 = data.numeric_class(alpha_test_y)
y2 = data.numeric_class(alpha_predict_y)

#print(eval.confusion(alpha_predict_y, alpha_test_y, classes=26))
#print(sklearn.metrics.accuracy_score(y1, y2))
#print(sklearn.metrics.precision_score(y1, y2))
#print(sklearn.metrics.recall_score(y1, y2))

''' predict one real image'''
test_img = data.preprocess_image(c.GWEN_W)
print(predict_az(test_img))


