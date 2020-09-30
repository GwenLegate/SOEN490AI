# This script contains the training parameters and algorithms used to train the ASL alphabet model

# Import statements
from utilities.data_processing import get_data

# Import data from source file
train_path = 'datasets/asl_alphabet/sign_mnist_train.csv'
train_data = get_data(train_path)

