# This script tests the performance of the trained model and creates statistical plots for visualization

# Import statements
from utilities.data_processing import get_data

# Import data from source file
test_path = 'datasets/asl_alphabet/sign_mnist_test.csv'
test_data = get_data(test_path)
