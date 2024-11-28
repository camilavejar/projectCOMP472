# Project COMP 472
Goal: develop and compare 4 AI models on a part of CIFAR-10 dataset.

## Data Preprocessing
First start by obtaining and preprocessing the data.

1- Run the file dataPreprocessing.py

This file will take care of getting the cifar10 dataset, unpickle it to get the images, sort the images by label and get the first 500 images/label in training and the first 100 images/label for training. You will have a Sorted dataset with all the images. This file will also extract the feature vectors using RESNET18. The images are resized and normalized before passing through the model. The output are four numpy files containing the feature vectors for both features and their labels for training and testing.

 ## Training the models (Naive Bayes, decision-tree, MLP)

All models have been saved to be easily loaded and evaluated. However, here are the files used to train them. The model files are in the models folder.

### 1- Running naiveBayesClassifier.py

 This file will load the numpy data saved in the preprocessing step. It will then use PCA to reduce the feature vector size from 512 to 50. The data can then pass through the python naive Bayes and print the accuracy. It also passes through the sklearn naive bayes and prints the accuracy. Note that the code to save the models is commented out because the repo already has the models saved.

### 2- Running decisionTreeClassifier.py

 This file will load the numpy data saved in the preprocessing step. It will then use PCA to reduce the feature vector size from 512 to 50. The data can then pass through the Python decision tree and print the accuracy. The training of two extra models with different max depths is also included to compare during evaluation. It also passes through the sklearn decision tree and prints the accuracy. Note that the code to save the models is commented out because the repo already has the models saved.
 
### 3- Running decisionTreeClassifier.py

 This file will load the numpy data saved in the preprocessing step. It will then use PCA to reduce the feature vector size from 512 to 50. It will then take this data and transform in into pytorch tensors because numpy isn't compatible with the model here. Once the data is ready, it will pass through the MLP model and then save it. Note that here, I did not create multiple models, but instead modified the layers and ran the code to get the new versions. The changes are explained in the layers as comments. All the models are saved to the repo.

 ## Evaluation (Naive Bayes, decision-tree, MLP)
 The evaluation file will display the confusion matrices for all models as well as the metric table comparing them. 

 ### 1- Running evaluation.py
 This file will load the test data, load all the saved models and evaluate them by displaying the accuracies as well as their confusion matrices. At the end, there will be the metric table.
 
Here is a sample confusion matrix:

![image](https://github.com/user-attachments/assets/64445fe0-f494-403b-b12a-2dfabc873d76)

Here is the output metric table:

![image](https://github.com/user-attachments/assets/bbaaec2d-33f7-42e7-9cc1-87a71c3e17e4)

## CNN model 
This model was too heavy for me to run on my laptop and so I used Google collab apart from the other models.
### 1. Running CNNClassifier.ipynb
 I included the jupyter notebook will all the steps needed to be able to run it on Colab.
 It will get the data as images in a smaller dataset like in the preprocessing file but it will not pass it through the restnet18 model. The file has the model and you can run the next cell to train the model and save it. Like in the MLP, I did not create multiple instances of the model to save but instead just modified the layers and reran the model to save it. The saved models can be loaded by running the imports and data cell. There will be displayed the accuracies of each model with confusion matrices as well as a final metric table like the one present for the other models.

Here is a sample confusion matrix:

![image](https://github.com/user-attachments/assets/b2f4070d-4cd8-41c8-a985-4d0dd78bf4de)

Here is the output metric table:

![image](https://github.com/user-attachments/assets/ead01974-c906-41cd-afbd-c7798fe489dc)

The saved models will be included in a different place because they are too big to upload.
Here is a google drive with the models: https://drive.google.com/drive/folders/1Q_JEIfWdAAhOpw4JhRfsMpbaYq9CSbeB?usp=sharing










 
