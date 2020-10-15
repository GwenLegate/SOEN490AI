import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import utilities.data_processing as data
import constants as c

'''Flags to control execution'''
CHECK_BALANCE = False
TRAIN = True

''' Check for GPU, if no GPU, use CPU'''
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

''' Define hyper parameters'''
KERNEL = 5
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 72

'''input parameters'''
train_img_xdim, train_img_ydim = 28, 28

''' Import data from source file and convert to PyTorch tensors'''
alphabet_train_data = data.get_data(c.FILE_TRAIN_ALPHABET)
alphabet_trainX, alphabet_trainy = alphabet_train_data[:, 1:], alphabet_train_data[:, 0]

'''reformat y to one-hot-vector-format for comparison with outputs'''
reformat_y = np.zeros((len(alphabet_trainy), 26))
i = 0
while i < len(alphabet_trainy):
    reformat_y[i][alphabet_trainy[i]]=1
    i +=1

alphabet_trainy = reformat_y

'''put numpy arrays into PyTorch tensors'''
alphabet_inputs = torch.from_numpy(alphabet_trainX).view(-1, train_img_xdim, train_img_xdim)
'''normalize pixel values'''
alphabet_inputs = alphabet_inputs/255.0
alphabet_labels = torch.from_numpy(alphabet_trainy)

'''Check balance of dataset'''
sign_totals = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0, 21:0, 22:0, 23:0, 24:0, 25:0}

if(CHECK_BALANCE):
   data.check_balance(sign_totals, alphabet_trainy)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, KERNEL)
        self.conv2 = nn.Conv2d(32, 32, KERNEL)
        self.conv3 = nn.Conv2d(32, 64, KERNEL)

        self.fc1 = nn.Linear(64*3*3, 512) # flattens cnn output
        self.fc2 = nn.Linear(512, 26)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # drops out couple of random neurons in the neural network to avoid overfitting
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1, 3*3*64 )  # .view is reshape, this flattens X for the linear layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)  # this is output layer. No activation.
        return F.softmax(x, dim=1)

alphabet_cnn = Net().to(device)
optimizer = optim.Adam(alphabet_cnn.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

''' training method'''
def train(alphabet_cnn):
    for epoch in range(EPOCHS):
        for i in range(0, len(alphabet_inputs), BATCH_SIZE):
            batch_x = alphabet_inputs[i:i + BATCH_SIZE].view(-1, 1, train_img_xdim, train_img_ydim)
            batch_y = alphabet_labels[i:i + BATCH_SIZE]
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            alphabet_cnn.zero_grad()
            outputs = alphabet_cnn(batch_x)
            loss = loss_fn(outputs, batch_y.float())
            loss.backward()
            optimizer.step()

''' set TRAIN=True to train model, learned weights are serialized and saved to the 'trained_models' directory'''
if(TRAIN):
    train(alphabet_cnn)
    torch.save(alphabet_cnn.state_dict(), c.MODEL_SAVE_PATH+"/alphabet_model.pt")
