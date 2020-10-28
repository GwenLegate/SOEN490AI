import os
FILE_TRAIN_ALPHABET = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+ \
                      "/SOEN490AI/datasets/asl_alphabet/sign_mnist_train.csv"
FILE_TEST_ALPHABET = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + \
                     "/SOEN490AI/datasets/asl_alphabet/sign_mnist_test.csv"
FILE_DIGIT_FEATURES = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+ \
                      "/SOEN490AI/datasets/asl_digits/X.npy"
FILE_DIGIT_LABELS = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+ \
                      "/SOEN490AI/datasets/asl_digits/Y.npy"
MODEL_SAVE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + \
                     "/SOEN490AI/trained_models"

