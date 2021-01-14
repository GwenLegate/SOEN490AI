import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
import numpy as np
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import utilities.data_processing as data
import constants as c

# Flags to control execution
TRAIN = True
CONTINUE_TRAINING = True
LOAD = True

# Check for GPU, if no GPU, use CPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

# Define hyper parameters
LEARNING_RATE = 0.0001
EPOCHS = 25

BATCH_SIZE = 50

# input parameters
MODEL_NAME = f"alpha_model-{int(time.time())}" # make logfile (time.time() gives a time stamp so you can have a history)

sign_totals = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0,
               19:0, 20:0, 21:0, 22:0, 23:0, 24:0, 25:0}

# load training and testing data and put them into torch tensors
if LOAD:
    data_X = data.get_training_arr("alpha_train_features_scale_shuffle.npy")
    data_y = data.get_training_arr('alpha_train_labels_shuffle.npy')

    alpha_X_validate, alpha_X = data_X[:14400, :], data_X[14400:, :]
    alpha_y_validate, alpha_y = data_y[:14400, :], data_y[14400:, :]
    print(alpha_X.shape, alpha_y.shape)
    print(alpha_X_validate.shape, alpha_y_validate.shape)

    alpha_X = torch.from_numpy(alpha_X).type('torch.FloatTensor')
    alpha_y = torch.from_numpy(alpha_y)

    alpha_X_validate = torch.from_numpy(alpha_X_validate).type('torch.FloatTensor')
    alpha_y_validate = torch.from_numpy(alpha_y_validate)

# Define CNN parameters
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=2)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2)
        self.conv4 = nn.Conv2d(64, 88, kernel_size=3, stride=1, padding=2)

        self.avgpool = nn.AdaptiveAvgPool2d(3)
        self.fc1 = nn.Linear(88*3*3, 512) # flattens cnn output
        self.fc2 = nn.Linear(512, 26)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.cpu()
        x = x.detach().numpy()
        np.save('visualize_activation1.npy', x)
        x = torch.from_numpy(x).type('torch.FloatTensor').to(device)

        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.cpu()
        x = x.detach().numpy()
        np.save('visualize_activation2.npy', x)
        x = torch.from_numpy(x).type('torch.FloatTensor').to(device)

        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        # drops out couple of random neurons in the neural network to avoid overfitting
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.avgpool(x))

        x = x.view(-1, 3*3*88)  # .view is reshape, this flattens X for the linear layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
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
    y = torch.from_numpy(data.numeric_class(y.cpu().numpy()))
    y = y.to(device)
    loss = loss_fn(outputs, y.long())
    if train:
        loss.backward()
        optimizer.step()
    return accuracy, loss

alphabet_cnn = Net().to(device)
# load current model if you want to continue to build off it
if CONTINUE_TRAINING:
    print("previous model loaded")
    alphabet_cnn.load_state_dict(torch.load(c.MODEL_SAVE_PATH + "/alphabet_model.pt", map_location=device))
    #summary(alphabet_cnn, (10, 200, 200))

optimizer = optim.Adam(alphabet_cnn.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

'''tests accuracy and loss on a random slice of the test data.
# size: the amount of test instances to use.
# returns the accuracy and loss for the test data being fed through the model'''
def test(size):
    random_start = np.random.randint(len(alpha_X_validate) - size)
    X, y = alpha_X_validate[random_start:random_start + size], alpha_y_validate[random_start:random_start + size]
    with torch.no_grad():
        test_accuracy, test_loss = feed_model(X.view(-1, 1, 200, 200).to(device), y.to(device))
    return test_accuracy, test_loss

# training method, includes a log file to track training progress
def train():
    global LEARNING_RATE
    NUM_BATCH = 100 # don't want to test every pass, set "NUM_BATCH"  to test every NUM_BATCH pass
    with open("alphabet_model.log", "a+") as f:
        init_time = time.time()
        for epoch in range(EPOCHS):
            print(epoch)

            if epoch == 10 or epoch == 20:
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