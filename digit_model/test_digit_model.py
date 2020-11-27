import torch
import sklearn
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import utilities.data_processing as data
import utilities.evaluation_metrics as eval
import constants as c
from digit_model.train_digit_model import Net

''' check for GPU, if no GPU, use CPU '''
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

'''testing'''
digit_cnn = Net()
digit_cnn.load_state_dict(torch.load(c.MODEL_SAVE_PATH + "/digit_model.pt", map_location=device))
digit_test_y = data.get_training_arr("digit_test_labels.npy")
digit_test_X = data.get_training_arr("digit_test_features.npy")

print(digit_test_y.shape, digit_test_X.shape)


''' returns predictions from the model.  The default type 1 will return the predicted letter, type 2 will return a 
one-hot-vector prediction for the inputs that can be used for comparison with the labels'''

def predict_digit(input, type=1):
    try:
        w, l = input.shape
    except ValueError:
        _, w, l = input.shape

    input_tensor = torch.from_numpy(input).view(-1, w, l).type('torch.FloatTensor')
    predict_vect = digit_cnn(input_tensor.view(-1, 1, w, l)).detach().numpy()
    if type == 1:
        predict_val = np.argmax(predict_vect)
        return predict_val
    if type == 2:
        return predict_vect

digit_predictions = predict_digit(digit_test_X, type=2)


y1 = data.numeric_class(digit_test_y)
y2 = data.numeric_class(digit_predictions)

print(sklearn.metrics.accuracy_score(y1, y2))
print(sklearn.metrics.precision_score(y1, y2, average="macro"))
print(sklearn.metrics.recall_score(y1, y2, average="macro"))
print(sklearn.metrics.confusion_matrix(y1, y2))

''' predict one real image'''
test_img = data.preprocess_image(c.GWEN5)
print(predict_digit(test_img))


