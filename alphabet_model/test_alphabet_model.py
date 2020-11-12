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

''' check for GPU, if no GPU, use CPU '''
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

'''global variables and flags to control execution'''
PLOT_ACCURACY_LOSS = False
REPROCESS_TEST_IMAGES = False
RECALCULATE_TEST_PREDICTIONS = False

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

''' image process on generic training set '''
if REPROCESS_TEST_IMAGES:
    # preprocess images and add them to the input feature array
    alpha_test_X = data.get_data('test_inputs.npy')
    num = 0
    for root, dirs, files in os.walk(c.TEST_ALPHABET_IMGS_BASEDIR):
        for name in files:
            px = data.preprocess_image(os.path.join(root, name))
            r, c, = px.shape

            if r == 200 and c == 200:
                px = px.reshape(-1, 200, 200)
                alpha_test_X = np.append(alpha_test_X, px, axis=0)
            else:
                # pad image
                add_r = 200 - r
                add_c = 200 - c
                if not add_r == 0:
                    px.vstack((np.zeros(add_r, c)), px)
                if not add_c == 0:
                    px.hstack((np.zeros((200, add_c)), px))
                px = px.reshape(-1, 200, 200)
                alpha_test_X = np.append(alpha_test_X, px, axis=0)
    np.save('test_inputs.npy', alpha_test_X)

    # all letters have 30 instances except for j and z which have 0.  Create y and put into one hot vector format
    alpha_test_y = np.zeros(30, dtype=int)
    for i in range(1, 25):
        if not i == 9:
            arr = np.full(30, i)
            alpha_test_y = np.concatenate((alpha_test_y, arr))

    alpha_test_y = data.one_hot_vector(alpha_test_y, num_classes=26)
    np.save('test_labels.npy', alpha_test_y)

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
    alpha_test_X = alpha_test_X.reshape(720, -1)
    alpha_test_y = data.numeric_class(alpha_test_y).reshape(-1, 1)
    xy = np.hstack((alpha_test_X, alpha_test_y))
    np.random.shuffle(xy)
    alpha_test_X, alpha_test_y = xy[:, :40000].reshape(-1, 200, 200), xy[:, -1]
    alpha_test_y = data.one_hot_vector(alpha_test_y.astype(int), num_classes=26)
    np.save('test_labels.npy', alpha_test_y) # save shuffled order so the saved y corresponds to the predicted y
    alpha_predict_y = predict_az(alpha_test_X, type=2)
    np.save('test_predictions.npy', alpha_predict_y)

''' load test predictions'''
alpha_predict_y = data.get_data("test_predictions.npy")

y1 = data.numeric_class(alpha_test_y)
y2 = data.numeric_class(alpha_predict_y)

print(eval.confusion(alpha_predict_y, alpha_test_y, classes=26))
print(sklearn.metrics.accuracy_score(y1, y2))
print(sklearn.metrics.precision_score(y1, y2))
print(sklearn.metrics.recall_score(y1, y2))

''' predict one real image'''
test_img = data.preprocess_image(c.Gwen_W)
print(predict_az(test_img))


