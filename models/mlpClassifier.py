from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        # Initialize the Multi-Layer Perceptron
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            # 1st hidden layer: 50 input features, 512 output features
            nn.Linear(input_size, hidden_size),  
            nn.ReLU(),  # ReLU activation 

            # 2nd hidden layer: 512 input features, 512 output features
            nn.Linear(hidden_size, hidden_size),  
            nn.BatchNorm1d(hidden_size),  # Batch Normalization 
            nn.ReLU(),  # ReLU activation 

            # ADDED 3rd hidden layer: 512 input features, 512 output features
            # nn.Linear(hidden_size, hidden_size),  
            # nn.ReLU(),  # ReLU activation 

            ## ADDED Dropout layer
            # nn.Dropout(p=0.5),  # Add dropout for regularization

            # Output layer: 512 input features, 10 output features
            nn.Linear(hidden_size, num_classes), 
        )

    def forward(self, x):
        # Forward pass through the MLP, already defined in the layers
        return self.layers(x)

if __name__ == "__main__":
    # Load training and testing data
    features_train = np.load('features_trainFullsizeVectors.npy')  # Features for training
    labels_train = np.load('labels_trainFullsizeVectors.npy')  # Labels for training
    features_test = np.load('features_testFullsizeVectors.npy')    # Features for testing
    labels_test = np.load('labels_testFullsizeVectors.npy')    # Labels for testing
    # Apply PCA to reduce the dimensionality of the feature vectors from 512 to 50
    pca = PCA(n_components=50)
    features_train = pca.fit_transform(features_train)
    features_test = pca.transform(features_test)

    # Convert to PyTorch tensors because we can't use NumPy arrays directly
    X_train = torch.tensor(features_train, dtype=torch.float32)
    y_train = torch.tensor(labels_train, dtype=torch.long)
    X_test = torch.tensor(features_test, dtype=torch.float32)
    y_test = torch.tensor(labels_test, dtype=torch.long)

    # Create data loaders for training and testing to use in PyTorch
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    ''' MLP Classifier '''
    # Initialize the MLP model
    model = MLP(input_size=50, hidden_size=512, num_classes=10)
    criterion = nn.CrossEntropyLoss()  # Define the loss function
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # Define the optimizer
    
    # Training loop
    epochs = 20
    for epoch in range(epochs):
        # Initialize the model for training
        model.train()  
        running_loss = 0.0
        correctlyClassified = 0
        totalSamples = 0
        
        for features, labels in train_loader:
            # Zero the gradients
            optimizer.zero_grad() 
            predictions = model(features)  # Forward pass
            loss = criterion(predictions, labels)  # Compute the loss with sgd
            loss.backward()  # Backward pass 
            optimizer.step()  # Update weights
            running_loss += loss.item()
            
            # Compute accuracy
            _, predicted = torch.max(predictions, 1)
            totalSamples += labels.size(0)
            correctlyClassified += (predicted == labels).sum().item()
        # Print the training accuracy and loss
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correctlyClassified / totalSamples:.2f}%")

    print('Finished Training')
    
    #Evaluate the model on the test set and get accuracy
    model.eval()  
    correctlyClassified = 0
    totalSamples = 0
    # Disable gradient tracking because we don't need it for evaluation
    with torch.no_grad(): 
        for features, labels in test_loader:
            predictions = model(features)
            _, predicted = torch.max(predictions, 1)
            totalSamples += labels.size(0)
            correctlyClassified += (predicted == labels).sum().item()
    
    print(f"Test Accuracy: {100 * correctlyClassified / totalSamples:.2f}%")

    # Saving the model
    torch.save(model, 'MLPclassifier.pth')
    # torch.save(model, 'MLPclassifierAddDropOut.pth')
    # torch.save(model, 'MLPclassifierAddLayer.pth')
    # torch.save(model, 'MLPclassifierAddLayerAndDropOut.pth')


