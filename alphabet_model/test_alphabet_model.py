import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utilities.evaluation_metrics import *
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
        self.conv2 = nn.Conv2d(16, 24, kernel_size=5, stride=1, padding=3)
        self.conv3 = nn.Conv2d(24, 32, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=2)
        self.conv6 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2)


        self.avgpool = nn.AdaptiveAvgPool2d(3)
        self.fc1 = nn.Linear(64*3*3, 512) # flattens cnn output
        self.fc2 = nn.Linear(512, 26)

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
alphabet_cnn = Net()
alphabet_cnn.load_state_dict(torch.load(c.MODEL_SAVE_PATH + "/alphabet_model.pt", map_location=device))
alphabet_cnn.cuda()
alphabet_cnn.to('cpu') # puts model on cpu

alpha_key = {0:"A", 1:"B", 2:"C", 3:"D", 4:"E", 5:"F", 6:"G", 7:"H", 8:"I", 9:"J", 10:"K", 11:"L", 12:"M", 13:"N",
             14:"O", 15:"P", 16:"Q", 17:"R", 18:"S", 19:"T", 20:"U", 21:"V", 22:"W", 23:"X", 24:"Y", 25:"Z"}

# checks if two models have equal weights
def check_equal(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True

''' returns predictions from the model.  The default type 1 will return the predicted letter, type 2 will return a 
one-hot-vector prediction for the inputs that can be used for comparison with the labels'''

def predict_az(input, type=1):
    try:
        w, l = input.shape
    except ValueError:
        _, w, l = input.shape

    input_tensor = torch.from_numpy(input).view(-1, w, l).type('torch.FloatTensor').to(device)
    predict_vect = alphabet_cnn(input_tensor.view(-1, 1, w, l))
    predict_vect = predict_vect.cpu()
    predict_vect = predict_vect.detach().numpy()
    if type == 1:
        predict_val = np.argmax(predict_vect)
        return alpha_key[predict_val], predict_vect
    if type == 2:
        return predict_vect

# use this method to test the images from the team dataset since each image is a different size and needs to be loaded,
# processed and predicted individually
def test_images_alphabet():
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    confusion = np.zeros((24, 24), dtype=int)
    for letter in letters:
        print('testing ' + letter)
        for root, dirs, files in os.walk(c.TEAM_ALPHABET_IMGS_BASEDIR + letter):
            for name in files:
                px = preprocess_image(os.path.join(root, name))
                predict, _ = predict_az(px)
                num = confusion[letters.index(predict), letters.index(letter)] + 1
                confusion[letters.index(predict), letters.index(letter)] = num
    # accuracy and percision calcs
    num_correct = 0
    precision = np.zeros(24)
    for i in range(24):
        num_correct += confusion[i, i]
        if np.sum(confusion, axis=1)[i] == 0:
            precision[i] = 0
        else:
            precision[i] = confusion[i,i] / np.sum(confusion, axis=1)[i]
    accuracy = num_correct / np.sum(confusion)
    precision = np.mean(precision)
    print('accuracy: ' + str(accuracy))
    print('precision: ' + str(precision))
    print(confusion)

# use this method to test on a dataset with same sized images that can be processed as a batch and loaded from a saved
# numpy array where preprocessing has already been done to save time
def test(alpha_X_test, alpha_y_test):
    alpha_predict_y = predict_az(alpha_X_test.reshape(-1, 200, 200), type=2)
    #print(alpha_predict_y)

    y1 = numeric_class(alpha_y_test)
    y2 = numeric_class(alpha_predict_y)

    print(sklearn.metrics.accuracy_score(y1, y2))
    print(sklearn.metrics.precision_score(y1, y2, average='macro'))
    print(sklearn.metrics.recall_score(y1, y2, average='macro'))
    print(sklearn.metrics.confusion_matrix(y1, y2))

# use this method to test one image, arg is the path to an image file
def test_real_image(img_path):
    test_img = preprocess_image(img_path)
    res, vect = predict_az(test_img)
    print(res)
    print(vect)

# Select testing regime.
# SET = 1 corresponds to the noiseless dataset
# SET = 2 corresponds to testing with team dataset (takes a long time to test b/c the images are all different sizes)
# SET = any other number corresponds to the default option of the noisy set obtained online
SET = 1

if SET == 1:
    data_X = get_training_arr('alpha_train_features_no_noise_shuffled.npy')
    data_y = get_training_arr('alpha_train_labels_no_noise_shuffled.npy')

    alpha_X_test = data_X[:100, :]
    alpha_y_test = data_y[:100, :]

    test(alpha_X_test, alpha_y_test)

if SET == 2:
    test_images_alphabet()

else:
    alpha_X_test = get_training_arr('alpha_test_inputs.npy')
    alpha_y_test = get_training_arr('alphabet_test_labels.npy')

    test(alpha_X_test, alpha_y_test)
