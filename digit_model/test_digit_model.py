import sklearn
import numpy as np
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import utilities.data_processing as data
import constants as c


''' check for GPU, if no GPU, use CPU '''
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

#digit model class
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=3)
        self.conv2 = nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=2)
        self.conv3 = nn.Conv2d(24, 32, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2)
        self.conv6 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2)

        self.avgpool = nn.AdaptiveAvgPool2d(3)
        self.fc1 = nn.Linear(64*3*3, 512) # flattens cnn output
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = F.relu(F.max_pool2d(self.conv5(x), 2))
        x = F.relu(F.max_pool2d(self.conv5(x), 2))

        x = F.relu(self.avgpool(x))

        x = x.view(-1, 3*3*64)  # .view is reshape, this flattens X for the linear layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)  # this is output layer. No activation.
        return F.softmax(x, dim=1)


'''testing'''
digit_cnn = Net()
digit_cnn.load_state_dict(torch.load(c.MODEL_SAVE_PATH + "/digit_model.pt", map_location=device))
digit_test_y = data.get_training_arr("digit_labels_shuffled.npy")[:2776, :]
digit_test_X = data.get_training_arr("digit_features_shuffled.npy")[:2776, :]

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


