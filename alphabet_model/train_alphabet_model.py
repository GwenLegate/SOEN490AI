import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import utilities.data_processing as data
import constants as c

# Flags to control execution
CHECK_BALANCE = False
TRAIN = False
CONTINUE_TRAINING = False
PROCESS_MNIST = False
REPROCESS_TRAINING_IMAGES = True

# Check for GPU, if no GPU, use CPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

# Define hyper parameters
KERNEL = 5
LEARNING_RATE = 0.001
EPOCHS = 80
BATCH_SIZE = 100

# input parameters
MODEL_NAME = f"alpha_model-{int(time.time())}" # make logfile (time.time() gives a time stamp so you can have a history)

if PROCESS_MNIST:
    # Import data from MNIST source file and convert to numpy arrays for features and labels
    alphabet_train_data = data.get_data(c.FILE_TRAIN_ALPHABET)
    alphabet_test_data = data.get_data(c.FILE_TEST_ALPHABET)
    alphabet_testX, alphabet_testy = alphabet_test_data[:, 1:], alphabet_test_data[:, 0]
    alphabet_trainX, alphabet_trainy = alphabet_train_data[:, 1:], alphabet_train_data[:, 0]

    # reformat y to one-hot-vector-format for comparison with outputs
    alphabet_testy = data.one_hot_vector(alphabet_testy, num_classes=26)
    alphabet_trainy = data.one_hot_vector(alphabet_trainy, num_classes=26)

    # put numpy arrays into PyTorch tensors
    alphabet_train_features = torch.from_numpy(alphabet_trainX).view(-1, 28, 28)
    alphabet_test_features = torch.from_numpy(alphabet_testX).view(-1, 28, 28)
    # normalize pixel values of features
    alphabet_train_features = alphabet_train_features / 255.0
    alphabet_test_features = alphabet_test_features / 255.0

    alphabet_train_labels = torch.from_numpy(alphabet_trainy)
    alphabet_test_labels = torch.from_numpy(alphabet_testy)

alpha_X = data.get_training_arr('training_data.npy')
print(alpha_X.shape)

if REPROCESS_TRAINING_IMAGES:
    # preprocess images and add them to the input feature array
    for root, dirs, files in os.walk(c.TRAIN_ALPHABET_IMGS_BASEDIR+"K"):
        for name in files:
            print(name)
            px = data.preprocess_image(os.path.join(root, name)).reshape(-1, 200, 200)
            alpha_X = np.append(alpha_X, px, axis=0)
        alpha_X = StandardScaler().fit_transform(alpha_X.reshape(-1, 200 * 200))  # standardize data around mean = 0
        np.save('training_data.npy', alpha_X.reshape(-1, 200, 200))

    print(alpha_X.shape)

    #all letters have 3000 instances except J and Z which have 0.  Create y and put into one hot vector format
    alpha_y = np.zeros(3000, dtype=int)
    for i in range(1, 25):
        if not i == 9:
            arr = np.full(3000, i)
            alpha_y = np.concatenate((alpha_y, arr))
    alpha_y = data.one_hot_vector(alpha_y, num_classes=26)

    # standardize, shuffle and split training and testing data and put them into torch tensors
    alpha_X, alpha_X_test, alpha_y, alpha_y_test = train_test_split(alpha_X, alpha_y, test_size=0.15)
    alpha_X = torch.from_numpy(alpha_X).type('torch.FloatTensor')
    alpha_y = torch.from_numpy(alpha_y)
    alpha_X_test = torch.from_numpy(alpha_X_test).type('torch.FloatTensor')
    alpha_y_test = torch.from_numpy(alpha_y_test)

# Check balance of dataset
sign_totals = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0,
               19:0, 20:0, 21:0, 22:0, 23:0, 24:0, 25:0}

if CHECK_BALANCE:
   data.check_balance(sign_totals, alphabet_trainy)

# Define CNN parameters
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 6)
        self.conv2 = nn.Conv2d(32, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 3)

        self.avgpool = nn.AdaptiveAvgPool2d(3)
        self.fc1 = nn.Linear(64*3*3, 512) # flattens cnn output
        self.fc2 = nn.Linear(512, 26)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # drops out couple of random neurons in the neural network to avoid overfitting
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.avgpool(x))

        x = x.view(-1, 3*3*64 )  # .view is reshape, this flattens X for the linear layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)  # this is output layer. No activation.
        return F.softmax(x, dim=1)

# method to pass data through the model, set train to True if it is a training pass.
# Returns accuracy and loss of X, y passed
def feed_model(X, y, train=False):
    if train:
        alphabet_cnn.zero_grad()
    outputs = alphabet_cnn(X)
    compare = zip(outputs, y)
    num_correct = 0
    for n, m in compare:
        a = torch.argmax(n)
        b = torch.argmax(m)
        if a == b:
            num_correct += 1
    accuracy = num_correct/len(y)
    loss = loss_fn(outputs, y.float())
    if train:
        loss.backward()
        optimizer.step()
    return accuracy, loss

alphabet_cnn = Net().to(device)
# load current model if you want to continue to build off it
if CONTINUE_TRAINING:
    alphabet_cnn.load_state_dict(torch.load(c.MODEL_SAVE_PATH + "/alphabet_model.pt", map_location=device))

optimizer = optim.Adam(alphabet_cnn.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

'''tests accuracy and loss on a random slice of the test data.
# size: the amount of test instances to use.
# returns the accuracy and loss for the test data being fed through the model'''
def test(size):
    random_start = np.random.randint(len(alpha_X_test) - size)
    X, y = alpha_X_test[random_start:random_start + size], alpha_y_test[random_start:random_start + size]
    with torch.no_grad():
        test_accuracy, test_loss = feed_model(X.view(-1, 1, 200, 200).to(device), y.to(device))
    return test_accuracy, test_loss

# training method, includes a log file to track training progress
def train():
    NUM_BATCH = 100 # don't want to test every pass, set "NUM_BATCH"  to test every NUM_BATCH pass
    with open("alphabet_model.log", "a") as f:
        init_time = time.time()
        for epoch in range(EPOCHS):
            print(epoch)
            if epoch == 20 or epoch == 40 or epoch == 60:
                # save progress periodically in case we run out of time on the gpu
                torch.save(alphabet_cnn.state_dict(), c.MODEL_SAVE_PATH + "/alphabet_model.pt")
            for i in range(0, len(alpha_X), BATCH_SIZE):
                batch_x = alpha_X[i:i + BATCH_SIZE].view(-1, 1, 200, 200)
                batch_y = alpha_y[i:i + BATCH_SIZE]
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                train_accuracy, train_loss = feed_model(batch_x, batch_y, train=True) #train model with batch data

                if i % NUM_BATCH == 0:
                    test_accuracy, test_loss = test(size=100)
                    f.write(
                        f"{MODEL_NAME}, {round(time.time()-init_time, 4)}, {int(epoch)}, {round(float(test_accuracy), 5)}, {round(float(test_loss), 5)}, {round(float(train_accuracy), 5)}, {round(float(train_loss), 5)}\n")

# set TRAIN=True to train model, learned weights are serialized and saved to the 'trained_models' directory
if(TRAIN):
    train()
    torch.save(alphabet_cnn.state_dict(), c.MODEL_SAVE_PATH+"/alphabet_model.pt")