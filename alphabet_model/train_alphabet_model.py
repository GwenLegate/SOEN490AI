import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import utilities.data_processing as data
import constants as c

'''Flags to control execution'''
CHECK_BALANCE = False
TRAIN = False

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
EPOCHS = 10
BATCH_SIZE = 64

'''input parameters'''
train_img_xdim, train_img_ydim = 32, 32

''' Import data from source file and convert to PyTorch tensors'''
alphabet_train_data = data.get_data(c.FILE_TRAIN_ALPHABET)
alphabet_trainX, alphabet_trainy = alphabet_train_data[:, 1:], alphabet_train_data[:, 0]
alphabet_inputs = torch.from_numpy(alphabet_trainX).view(-1, 32, 32)
'''normalize pixel values'''
alphabet_inputs = alphabet_inputs/255.0
alphabet_labels = torch.from_numpy(alphabet_trainy, dtype=torch.float32)

'''Check balance of dataset'''
sign_totals = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0, 21:0, 22:0, 23:0, 24:0, 25:0}

if(CHECK_BALANCE):
   data.check_balance(sign_totals, alphabet_trainy)

'''Net class defines the architecture of the neural net, how data is fed through and error is back propagated'''
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, KERNEL)
        self.conv2 = nn.Conv2d(32, 64, KERNEL)
        self.conv3 = nn.Conv2d(64, 128, KERNEL)

        x = torch.randin(train_img_xdim, train_img_ydim).view(-1, 1, train_img_xdim, train_img_ydim)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512) # flattens cnn output
        self.fc2 = nn.Linear(512, 26)

    def convs(self, x):
        # max pooling over 2x2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # bc this is our output layer. No activation here.
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
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            print("loss "+loss)
''' set TRAIN=True to train model, learned weights are serialized and saved to the 'trained_models' directory'''
if(TRAIN):
    train(alphabet_cnn)
    torch.save(alphabet_cnn.state_dict(), c.MODEL_SAVE_PATH+"/alphabet_model.pt")