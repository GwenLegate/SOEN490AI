import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utilities.data_processing import *

''' check for GPU, if no GPU, use CPU '''
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
device = torch.device("cpu")

# alphabet model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=3)
        self.conv2 = nn.Conv2d(16, 24, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=2)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2)

        self.avgpool = nn.AdaptiveAvgPool2d(3)
        self.fc1 = nn.Linear(64*3*3, 512) # flattens cnn output
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = F.relu(F.max_pool2d(self.conv5(x), 2))
        x = F.relu(F.max_pool2d(self.conv6(x), 2))

        x = F.relu(self.avgpool(x))

        x = x.view(-1, 3*3*64)  # .view is reshape, this flattens X for the linear layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)  # this is output layer. No activation.
        return F.softmax(x, dim=1)

'''testing'''
digit_cnn = Net()
digit_cnn.load_state_dict(torch.load(c.MODEL_SAVE_PATH + "/digit_model.pt", map_location=device))
digit_cnn.cuda()
digit_cnn.to('cpu') # puts model on cpu

''' returns predictions from the model.  The default type 1 will return the predicted letter, type 2 will return a 
one-hot-vector prediction for the inputs that can be used for comparison with the labels'''

def predict_az(input, type=1):
    try:
        w, l = input.shape
    except ValueError:
        _, w, l = input.shape

    input_tensor = torch.from_numpy(input).view(-1, w, l).type('torch.FloatTensor').to(device)
    predict_vect = digit_cnn(input_tensor.view(-1, 1, w, l))
    predict_vect = predict_vect.cpu()
    predict_vect = predict_vect.detach().numpy()
    if type == 1:
        return np.argmax(predict_vect)

    if type == 2:
        return predict_vect

''' load testing X and y'''
data_X = get_training_arr('digit_features_shuffled.npy')
data_y = get_training_arr('digit_labels_shuffled.npy')

digit_X_test = data_X[:1000, :]
alpha_y_test = data_y[:1000, :]

alpha_predict_y = predict_az(digit_X_test.reshape(-1, 200, 200), type=2)
#print(alpha_predict_y)

y1 = numeric_class(alpha_y_test)
y2 = numeric_class(alpha_predict_y)

print(sklearn.metrics.accuracy_score(y1, y2))
print(sklearn.metrics.precision_score(y1, y2, average='macro'))
print(sklearn.metrics.recall_score(y1, y2, average='macro'))
print(sklearn.metrics.confusion_matrix(y1, y2))


''' predict one real image'''
test_img = preprocess_image(c.GWEN5)
#print(test_img)
'''x = get_training_arr('x.npy')
test_img = x[0]
test_img = preprocess_image(test_img)'''

'''plt.imshow((test_img * 255), cmap='gray')
plt.show()'''


result = predict_az(test_img)
print(result)

