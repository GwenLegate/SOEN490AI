# SOEN490AI

This repo contains code used to train AI models for use with the In Plain Sight application being developed for SOEN490
https://github.com/RjDrury/SOEN490

We will be using google colab to have access to a GPU to speed up the processing time required for training.
The python scripts required for training will be created outside of the google colab environment and pushed to the 
SOEN490AI repository.  The contents of the repository will them be cloned into a Google drive so they can be executed
by Google colab using the jupyter notebook "execute_ai_training.ipynb" located in the shared SOEN490 Google drive   


### Set up PyTorch Environment:

go to https://pytorch.org under the "Quick start locally" header select your configuration and copy the command that is 
generated in the "run this command" space.

Run the command copied in the previous step in the console. running this command should automatically install 
numpy, future, pillow and torchvision.
If you find that you are missing any of these libraries, install them manually using pip install

Next cd utilities and execute <python version.py> to verify your installation- it should print the installed version of 
PyTorch to the console

### Installing other required dependencies
cd to the SOEN490AI directory and execute <pip install -r requirements.txt> to install other required dependencies
for the project

### Dataset access 
download required datasets to the proper directory in the datasets directory (the directory structure exists but 
they are empty) you will need them when you are working on the scripts but they are too large to push to the 
repository.


## Model(s):

* ASL alphabet model to recognise the letters of the alphabet 
Dataset being used in the training: https://www.kaggle.com/datamunge/sign-language-mnist
