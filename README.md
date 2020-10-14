# SOEN490AI

This repo contains code used to train AI models for use with the In Plain Sight application being developed for SOEN490
https://github.com/RjDrury/SOEN490

We will be using google colab to have access to a GPU to speed up the processing time required for training.
The python scripts required for training will be created outside of the google colab environment and pushed to the 
SOEN490AI repository.  The contents of the repository will them be cloned into a Google drive so they can be executed
by Google colab using the jupyter notebook "execute_ai_training.ipynb" located in the shared SOEN490 Google drive   


### Set up PyTorch Environment:

go to https://pytorch.org under the "Quick start locally" header select your configuration (you will nee CUDA to make 
your verion of PyTorch usable on an GPU) and copy the command that is generated in the "run this command" space.

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

### Working with the GPU
to set PyTorch up to wot=rk on a GPU, you will need CUDA, here are the relevant links.
CUDA toolkit: https://developer.nvidia.com/cuda-toolkit
cudnn: https://developer.nvidia.com/cudnn (download, extract and move the bin, lib, include directories to your CUDA 
toolkit directory )

* Google colab works via a Jupyter notebook in your Google drive, the notebook is configured to pull the SOEN490AI 
repository into the drive where the dataset will need to be manually added and then the code can be executed


## Model(s):

* ASL alphabet model to recognise the letters of the alphabet 
Dataset being used in the training: https://www.kaggle.com/datamunge/sign-language-mnist
The training data consists of 27,455 cases of 28x28 pixel images with grayscale values between 0-255.  The data is 
pre-labeled so this model will be trained using supervised learning.  There are 7172 test cases that will be used to 
tune the fit of the model prior to attempting to evaluate it's performance on real world images.
Because the training images are 28x28 pixels, a convolutional neural network will be used to so the size of the input 
images passed from the In Plain Sight App will not be constrained to match this image size\[1\].  Image pre-processing 
should be applied to images passed from In Plain Sight since this will improve the accuracy of the model

** References

\[1\] Sumit Saha, "A Comprehensive Guide to Convolutional Neural Networks — the ELI5 way." Towards Data Science.com. 
Updated 10-15-2018. \[Website\]. 
Available: https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53, 
Accessed on: 09-18-20