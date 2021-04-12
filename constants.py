import os
TRAIN_ALPHABET_IMGS_BASEDIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+ \
                      "/SOEN490AI/datasets/asl_alphabet/asl_alphabet_train/"
TEST_ALPHABET_IMGS_BASEDIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+ \
                      "/SOEN490AI/datasets/asl_alphabet/asl_alphabet_test/"
TEAM_ALPHABET_IMGS_BASEDIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+ \
                      "/SOEN490AI/datasets/asl_alphabet/alphabet_team_dataset/"
TRAIN_DIGIT_IMGS_BASEDIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+ \
                      "/SOEN490AI/datasets/asl_digits/asl_digits_train/"
TEAM_DIGIT_IMGS_BASEDIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+ \
                      "/SOEN490AI/datasets/asl_digits/digits_team_dataset/"
MODEL_SAVE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + \
                     "/SOEN490AI/trained_models"
ACTIVATION_LYR_1 = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + \
                     "/SOEN490AI/alphabet_model/visualize_activation1.npy"
ACTIVATION_LYR_2 = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + \
                   "/SOEN490AI/alphabet_model/visualize_activation2.npy"\
