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
TRAIN = False
CONTINUE_TRAINING = False
LOAD = True

# Check for GPU, if no GPU, use CPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

# Define hyper parameters
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 50

# input parameters
MODEL_NAME = f"digit_model-{int(time.time())}"

if LOAD:
    # load, training and testing data into torch tensors
    data_X = data.get_training_arr("digit_train_features_shuffled.npy")
    data_y = data.get_training_arr('digit_train_labels_shuffled.npy')

    digit_X_validate, digit_X, digit_X_test = data_X[2776:8411, :], data_X[8411:, :], data_X[:2776, :]
    digit_y_validate, digit_y, digit_y_test = data_y[2776:8411, :], data_y[8411:, :], data_y[:2776, :]

    # use this config to do a hyperparameter search
    # digit_X, digit_X_validate  = data_X[2776:8411, :], data_X[:2776, :]
    # digit_y, digit_y_validate = data_y[2776:8411, :], data_y[:2776, :]

    print(digit_X_validate.shape, digit_y_validate.shape)
    print(digit_X.shape, digit_y.shape)

    digit_train_X = torch.from_numpy(digit_X).type('torch.FloatTensor')
    digit_train_y = torch.from_numpy(digit_y)
    digit_test_X = torch.from_numpy(digit_X_validate).type('torch.FloatTensor')
    digit_test_y = torch.from_numpy(digit_y_validate)

# Define CNN for digit model
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
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = F.relu(F.max_pool2d(self.conv5(x), 2))
        x = F.relu(F.max_pool2d(self.conv5(x), 2))

        x = F.relu(self.avgpool(x))

        x = x.view(-1, 3*3*64)  # .view is reshape, this flattens X for the linear layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)  # this is output layer. No activation.
        return F.softmax(x, dim=1)


digit_cnn = Net().to(device)
# load current model if you want to continue to build off it
if CONTINUE_TRAINING:
    print("previous model loaded")
    digit_cnn.load_state_dict(torch.load(c.MODEL_SAVE_PATH + "/digit_model.pt", map_location=device))

# use Adam optimization and cross entropy loss
optimizer = optim.Adam(digit_cnn.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

# method to pass data through the model, set train to True if it is a training pass.
# Returns accuracy and loss of X, y passed
def feed_model(X, y, train=False):
    if train:
        digit_cnn.zero_grad()
    outputs = digit_cnn(X)
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
    #loss = loss_fn(outputs, y.float())
    if train:
        loss.backward()
        optimizer.step()
    return accuracy, loss

'''tests accuracy and loss on a random slice of the test data.
# size: the amount of test instances to use.
# returns the accuracy and loss for the test data being fed through the model'''
def test(size):
    random_start = np.random.randint(len(digit_test_X) - size)
    X, y = digit_test_X[random_start:random_start + size], digit_test_y[random_start:random_start + size]
    with torch.no_grad():
        test_accuracy, test_loss = feed_model(X.view(-1, 1, 100, 100).to(device), y.to(device))
    return test_accuracy, test_loss

def train():
    global LEARNING_RATE
    NUM_BATCH = 300 # don't want to test every pass, set "NUM_BATCH"  to test every NUM_BATCH pass
    with open("digit_model.log", "a+") as f:
        init_time = time.time()
        for epoch in range(EPOCHS):
            print(epoch)

            if epoch == 15 or epoch == 30:
                # save progress periodically in case we run out of time on the gpu
                torch.save(digit_cnn.state_dict(), c.MODEL_SAVE_PATH + "/digit_model.pt")
            for i in range(0, len(digit_train_X), BATCH_SIZE):
                batch_x = digit_train_X[i:i + BATCH_SIZE].view(-1, 1, 100, 100)
                batch_y = digit_train_y[i:i + BATCH_SIZE]
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                train_accuracy, train_loss = feed_model(batch_x, batch_y, train=True) #train model with batch data
                if i % NUM_BATCH == 0:
                    test_accuracy, test_loss = test(size=100)
                    f.write(
                        f"{MODEL_NAME}, {round(time.time()-init_time, 4)}, {int(epoch)}, {round(float(test_accuracy), 5)}, {round(float(test_loss), 5)}, {round(float(train_accuracy), 5)}, {round(float(train_loss), 5)}\n")

# set TRAIN=True to train model, learned weights are serialized and saved to the 'trained_models' directory
if(TRAIN):
    train()
    torch.save(digit_cnn.state_dict(), c.MODEL_SAVE_PATH+"/digit_model.pt")
