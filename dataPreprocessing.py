import torch
import torchvision.transforms as transforms, torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.models as models
import pickle
import numpy as np
import os
import cv2
from PIL import Image

''' Load the datatset'''
def downloadDataset(): 
    # Download the dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    return trainset, testset

''' Load and umpickle the dataset'''
## Cifar-10 dataset comes in batches that need to be combined and unpickled to form full dataset
def loadAndUnpickleBatch(path):
    with open(path, 'rb') as file:
        batch = pickle.load(file, encoding='bytes')
        images = batch[b'data']
        labels = batch[b'labels']
        # Reshape the images to 32x32x3 or else images will be distorted
        images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return images, np.array(labels)
#Training data is divided into 5 batches
def loadAllBatches():
    images = []
    labels = []
    for i in range(1, 6):
        path = f'./data/cifar-10-batches-py/data_batch_{i}'
        image_batch, label_batch = loadAndUnpickleBatch(path)
        images.append(image_batch)
        labels.append(label_batch)
        combined_images = np.concatenate(images)
        combined_labels = np.concatenate(labels)
    return combined_images, combined_labels
#Test data is in a single batch
def loadTestBatch():
    path = './data/cifar-10-batches-py/test_batch'
    images, labels = loadAndUnpickleBatch(path)
    return images, labels

''' Sort images and labels into folders '''
def sortData(images, labels, folder):
    os.makedirs(folder, exist_ok=True)
    # Create a folder for each label 0-9
    for label in range(10):
        label_dir = os.path.join(folder, f'label_{label}')
        os.makedirs(label_dir, exist_ok=True)
    # Save images into the corresponding label folder
    for idx, (image, label) in enumerate(zip(images, labels)):
        label_dir = os.path.join(folder, f'label_{label}')
        image_filename = os.path.join(label_dir, f'image_{idx}.png')
        cv2.imwrite(image_filename, image)
    print('Data sorted into folders')

''' Reduce size of dataset '''
def get_first_n_images_per_class(images, labels, n):
    selected_images = []
    selected_labels = []

    for label in range(10):
        class_indices = np.where(labels == label)[0][:n]
        selected_images.append(images[class_indices])
        selected_labels.append(labels[class_indices])
    return np.concatenate(selected_images), np.concatenate(selected_labels)

def save_cifar10_images_by_label(features, labels, output_folder, n=100):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    label_counts = {}
    
    for idx, (image, label) in enumerate(zip(features, labels)):
        # Convert label to integer to facilitate sorting
        label = int(label)
        # Skip if we've saved `n` images for this label
        if label_counts.get(label, 0) >= n:
            continue
        # Create a subfolder for the label if it doesn't exist
        label_folder = os.path.join(output_folder, str(label))
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)
        # Save the image with original index as filename
        filename = f"image_{idx}.png"
        image_path = os.path.join(label_folder, filename)
        Image.fromarray(image).save(image_path)
        label_counts[label] = label_counts.get(label, 0) + 1
        # If we've saved `n` images for each label, we're done
        if len(label_counts) == len(np.unique(labels)) and all(c >= n for c in label_counts.values()):
            break

    print(f"Saved {n} images per label in {output_folder}")


'''----------------------------------Preprocessing with Resnet18----------------------------------'''
def extractFeatureVectors(path, model, transform,type):
    dataset = datasets.ImageFolder(root=path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    features = []
    labels = []
    with torch.no_grad():  # Disable gradient tracking
        for image_tensor, label in dataloader:
            # Pass the inputs through resnet18 model
            outputs = model(image_tensor)
            # Flatten the outputs to a vector to easilu reduce the size of the feature vector
            outputs = outputs.view(outputs.size(0), -1) 
            # Add features and labels
            features.append(outputs)
            labels.append(label)    
    # Concatenate all features and labels to their lists
    features = torch.cat(features, dim=0)  
    labels = torch.cat(labels, dim=0)  

    # Convert PyTorch tensors to NumPy arrays because its easier to use with classifiers
    features = features.numpy()
    labels = labels.numpy()
    # Save the features and labels to reuse later
    np.save(f"features_{type}.npy", features)
    np.save(f"labels_{type}.npy", labels)
    print('Feature extraction completed')

if __name__ == '__main__':
    ## Download the dataset
    trainset, testset = downloadDataset()
    ### Load and sort the dataset
    # Training
    train_images, train_labels = loadAllBatches()
    # Test
    test_images, test_labels = loadTestBatch()

    ## Sort and Reduce size of dataset: training = 500/label, test = 100/label
    # Training
    save_cifar10_images_by_label(train_images, train_labels, "Sorted/Training", n=500)
    # Testing    
    save_cifar10_images_by_label(test_images, test_labels, "Sorted/Test", n=100)

    # Sorted dataset will be used as is for CNN model

    ### Preprocess the dataset for Resnet18
    # Defining the transformations to be applied to the images: Resizing, normalizing, tensor conversion

    # load the ResNet model
    newmodel = models.resnet18(pretrained=True)
    # Remove the last layer (FC layer) of the ResNet model
    newmodel = nn.Sequential(*list(newmodel.children())[:-1])
    newmodel.eval()  # Set to evaluation mode
    # Define the transformation (resize and normalize) to apply to images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Ensure images are resized to 224x224
        transforms.ToTensor(),
        # Using known mean and standard deviation values to normalize the images
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Extract features from the training and test datasets
    extractFeatureVectors("Sorted/Training", newmodel, transform, "trainFullsizeVectors")
    extractFeatureVectors("Sorted/Test", newmodel, transform, "testFullsizeVectors")
