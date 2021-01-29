# SOEN490AI

This repo contains code used to train AI models for use with the In Plain Sight application being developed for SOEN490
https://github.com/RjDrury/SOEN490

By the end of the capstone project the goal is to have models that will allow successful prediction of letters A-Z and 
numbers 0-9.  It has been difficult to achieve good accuracy with these models, so I am investigating two strategies 
independently to see which one yields the best results.
* training "master models" that are capable of returning the correct digit or letter given an image of a digit or a letter
* training individual one-vs-the-rest models capable of identifying if an image is or is not a particular letter or digit

## Set up PyTorch Environment:

go to https://pytorch.org under the "Quick start locally" header select your configuration (you will nee CUDA to make 
your verion of PyTorch usable on an GPU) and copy the command that is generated in the "run this command" space.

Run the command copied in the previous step in the console. running this command should automatically install 
numpy, future, pillow and torchvision.
If you find that you are missing any of these libraries, install them manually using pip install

Next cd utilities and execute <python version.py> to verify your installation- it should print the installed version of 
PyTorch to the console

## Working with the GPU
to set PyTorch up to work on a GPU, you will need CUDA, here are the relevant links.
CUDA toolkit: https://developer.nvidia.com/cuda-toolkit
cudnn: https://developer.nvidia.com/cudnn (download, extract and move the bin, lib, include directories to your CUDA 
toolkit directory )

## Installing other required dependencies
cd to the SOEN490AI directory and execute <pip install -r requirements.txt> to install other required dependencies
for the project

## Models:
* the training of the alphabet model was done in the train_alphabet_model.py script located in the alphabet_model directory
* testing and verification was done using the test_alphabet_model.py script also located in the alphabet_model directory
* attempts at training one-vs-the-rest models for both "A" and "W" were attempted but abandoned after showing strong 
  tendencies to default to the "NOT A" or "NOT W" options in both cases  
* the training of the digit model was done in the train_digit_model.py script located in the digit_model directory
* testing and verification was done using the test_digit_model.py script also located in the digit_model directory
* to get a better idea of how the model is behaving, I save copies of the first two layers and created a tool to view 
  their activations in evaluation_metrics.py in the utilities directory

## Alphabet model dataset
The basis of the alphabet dataset currently used to train the alphabet model is found on Kaggle under the header 
“ASL Alphabet”. It consists of 3000 instances of letters A-Z, each image is 200x200 pixels.  

### Known Limitations of the "ASL Alphabet" Dataset 
* Both the letter “T” and the letter “G” are incorrect
* Images are all taken with similar backgrounds using a small number of subjects
  
The inaccuracy of the letters “G” and “T” are the biggest concern and to remedy this, another dataset was obtained from 
Kaggle called “ASL and some words”. This dataset contains images for letters A-Z, numbers 0-9 and some words that 
can be made without motion.  There are 4000 images for each different sign. This new dataset provides some variety in 
terms of the image backgrounds and the amounts of subjects that submitted images that will be beneficial to model training. 
The owner of this dataset also realized that the “G” images in the original dataset were incorrect and he provided 4000 
new images with the correct sign for “G” so all 3000 “G” images able to be replaced. To keep the dataset balanced but 
also take advantage of the diversity present in the new images, I replaced 200-400 of each letter with images from the 
new set.

### Known Limitations of the "ASL and some words" Dataset
* many of the images in this dataset are reused from the "ASL Alphabet" dataset, limiting the number of new images available 
* Images need to be verified for quality since some of them are unusable. 
* Extra images for the letter “T” that are correct are provided but many of the images of "T" have the same error as the
"ASL Alphabet" dataset.  This means I was not able to find 3000 accurate images of the letter "T" for training, only 2114
  
## Digit Model Dataset
The original dataset used on the training of the digit model is called “Sign Language Digits Dataset”. It was obtained 
from github and contains 2180 images for digits from 0-9.  The “ASL and some words” dataset also contains images for 
numbers and any numbers whose quality was deemed usable were added to the dataset.  

Due to the variation in the numbers of usable images in the "ASL and some words" dataset, the numbers of images are not 
perfectly balanced but were kept close enough that this factor should not have a significant effect on the results.

## Pre-Processing of the image data
The script angels_image_processing_tool.py located in the image_processing directory contains code to sharpen the edges 
of the hand in the image, apply a gaussian blur technique to remove noise and greyscale the images.  The raw image 
should be fed to the method process_image(path).  A numpy array containing the pixel data after these methods are applied
is returned.

Machine learning performs better when the features are normalized and scaled. Since an important feature of this 
project is the size invariance of the images it accepts, normalization is done per image instead of per data set. First,
the pixel values are centered about the mean and then min max scaling is applied so that each feature value is a number 
0 and 1.

The scaling is applied in the proper order to each image using the method preprocess_image(img_path) in the 
data_processing.py script found in the utilities directory.  This method will also be used to preprocess images in the
proper order when they are passed in the In Plain Sight App.

## References
\[\1\]\ ASL Alphabet (The data set is a collection of images of alphabets from the American Sign Language, separated in 
29 folders which represent the various classes; accessed November 25, 2020). 
https://www.kaggle.com/grassknoted/asl-alphabet

\[\2\]\ ASL_and_some_words (images of letters,numbers and some words in ASL; accessed Jan 12 2021) 
https://www.kaggle.com/belalelwikel/asl-and-some-words

\[\3\]\ Sign Language Digits Dataset (Turkey Ankara Ayrancı Anadolu High School's Sign Language Digits Dataset accessed 
October 28, 2020) https://github.com/ardamavi/Sign-Language-Digits-Dataset

\[4\] Sumit Saha, "A Comprehensive Guide to Convolutional Neural Networks — the ELI5 way." Towards Data Science.com. 
Updated 10-15-2018. \[Website\]. 
Available: https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53, 
Accessed on: 09-18-20