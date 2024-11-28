import numpy as np
from sklearn.decomposition import PCA
import dill
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.mlpClassifier import MLP
from sklearn.metrics import precision_score, recall_score, f1_score


''' Load test data '''
def loadTestData():
    features_train = np.load('features_trainFullsizeVectors.npy')  # Features for training
    features_test = np.load('features_testFullsizeVectors.npy')    # Features for testing
    labels_test = np.load('labels_testFullsizeVectors.npy')    # Labels for testing
    pca = PCA(n_components=50)
    features_train = pca.fit_transform(features_train)
    features_test = pca.transform(features_test)
    return features_test, labels_test


''' Load models'''
def pickleLoadModel(path):
    with open(path, 'rb') as f:
        model = dill.load(f)
    return model

def torchLoadModel(path):
    model = torch.load(path)
    return model

''' Evaluate models '''
def evaluateModel(model, features_test, labels_test):
    y_pred = model.predict(features_test)
    # Evaluate accuracy
    accuracy = np.mean(y_pred == labels_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    # Generate the confusion matrix
    print(f"Confusion matrix:\n{np.array(confusion_matrix(labels_test, y_pred))}")

def evaluateTorchModel(model, features_test, labels_test):
    model.eval()  
    correctlyClassified = 0
    totalSamples = 0
    test_dataset = TensorDataset(torch.tensor(features_test, dtype=torch.float32), torch.tensor(labels_test, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # Evaluate accuracy
    for features, labels in test_loader:
        predictions = model(features)  # Forward pass
        _, predicted = torch.max(predictions, 1)
        totalSamples += labels.size(0)
        correctlyClassified += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correctlyClassified / totalSamples:.2f}%")
    # Generate the confusion matrix
    all_preds = []
    all_labels = []

    # Disable gradient tracking
    with torch.no_grad():
        for features, labels in test_loader:
            predictions = model(features)  # Forward pass
            _, predicted = torch.max(predictions, 1)  # Get class predictions
            all_preds.extend(predicted.cpu().numpy())  # Collect predictions
            all_labels.extend(labels.cpu().numpy())   # Collect true labels

    # Generate the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)

''' Metric table '''
def compute_metrics(y_true, y_pred):

    accuracy = np.mean(y_true == y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return accuracy, precision, recall, f1

def evaluate_and_store_metrics(model, features_test, labels_test, is_torch_model=False):

    if is_torch_model:
        model.eval()
        all_preds = []
        all_labels = []
        test_dataset = TensorDataset(torch.tensor(features_test, dtype=torch.float32),
                                      torch.tensor(labels_test, dtype=torch.long))
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        with torch.no_grad():
            for features, labels in test_loader:
                predictions = model(features)
                _, predicted = torch.max(predictions, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        y_pred = np.array(all_preds)
        y_true = np.array(all_labels)
    else:
        y_pred = model.predict(features_test)
        y_true = labels_test

    accuracy, precision, recall, f1 = compute_metrics(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Confusion Matrix": confusion
    }

if __name__ == "__main__":
    # Load data
    features_test, labels_test = loadTestData()
    
    # Load models
    gaussianNaiveBayesClassifier = pickleLoadModel('gaussianNaiveBayesClassifier.dill')
    gaussianNaiveBayesScikit = pickleLoadModel('gaussianNaiveBayesScikit.dill')

    decisionTreeClassifier50 = pickleLoadModel('decisionTreeClassifier50.dill')
    decisionTreeClassifier20 = pickleLoadModel('decisionTreeClassifier20.dill')
    decisionTreeClassifier80 = pickleLoadModel('decisionTreeClassifier80.dill')
    decisionTreeClassifierScikit = pickleLoadModel('decisionTreeClassifierScikit.dill')

    MLPclassifier = MLP(input_size=50, hidden_size=512, num_classes=10)
    MLPclassifier =torchLoadModel('MLPclassifier.pth')
    MLPclassifierAddLayerAndDropOut = MLP(input_size=50, hidden_size=512, num_classes=10)
    MLPclassifierAddLayerAndDropOut = torchLoadModel('MLPclassifierAddLayerAndDropOut.pth')
    MLPclassifierAddLayer = MLP(input_size=50, hidden_size=512, num_classes=10)
    MLPclassifierAddLayer = torchLoadModel('MLPclassifierAddLayer.pth')
    MLPclassifierDropOut = MLP(input_size=50, hidden_size=512, num_classes=10)
    MLPclassifierDropOut = torchLoadModel('MLPclassifierAddDropOut.pth')
   

    # Evaluate models get test accuracy and print confusion matrix
    print('For the Naive Bayes Classifiers:')
    print('Python Naive Bayes Classifier')
    evaluateModel(gaussianNaiveBayesClassifier, features_test, labels_test)
    print('Scikit\'s Naive Bayes Classifier')
    evaluateModel(gaussianNaiveBayesScikit, features_test, labels_test)

    print('\nFor the Decision Tree Classifiers:')
    print('Python Decision Tree Classifier with max depth 50')
    evaluateModel(decisionTreeClassifier50, features_test, labels_test)
    print('Python Decision Tree Classifier with max depth 20')
    evaluateModel(decisionTreeClassifier20, features_test, labels_test)
    print('Python Decision Tree Classifier with max depth 80')
    evaluateModel(decisionTreeClassifier80, features_test, labels_test)
    print('Scikit\'s Decision Tree Classifier')
    evaluateModel(decisionTreeClassifierScikit, features_test, labels_test)

    print('\nFor the MLP Classifiers:')
    print('MLP Classifier')
    evaluateTorchModel(MLPclassifier, features_test, labels_test)
    print('MLP Classifier with additional layer and dropout')
    evaluateTorchModel(MLPclassifierAddLayerAndDropOut, features_test, labels_test)
    print('MLP Classifier with additional layer')
    evaluateTorchModel(MLPclassifierAddLayer, features_test, labels_test)
    print('MLP Classifier with dropout')
    evaluateTorchModel(MLPclassifierDropOut, features_test, labels_test)

    # Evaluate models and store metrics
    results = {
        "Naive Bayes Classifier (Python)": evaluate_and_store_metrics(gaussianNaiveBayesClassifier, features_test, labels_test),
        "Naive Bayes Classifier (Scikit)": evaluate_and_store_metrics(gaussianNaiveBayesScikit, features_test, labels_test),
        "Decision Tree (Max Depth 50)": evaluate_and_store_metrics(decisionTreeClassifier50, features_test, labels_test),
        "Decision Tree (Max Depth 20)": evaluate_and_store_metrics(decisionTreeClassifier20, features_test, labels_test),
        "Decision Tree (Max Depth 80)": evaluate_and_store_metrics(decisionTreeClassifier80, features_test, labels_test),
        "Decision Tree (Scikit)": evaluate_and_store_metrics(decisionTreeClassifierScikit, features_test, labels_test),
        "MLP Classifier": evaluate_and_store_metrics(MLPclassifier, features_test, labels_test, is_torch_model=True),
        "MLP Classifier (Add Layer and Dropout)": evaluate_and_store_metrics(MLPclassifierAddLayerAndDropOut, features_test, labels_test, is_torch_model=True),
        "MLP Classifier (Add Layer)": evaluate_and_store_metrics(MLPclassifierAddLayer, features_test, labels_test, is_torch_model=True),
        "MLP Classifier (Dropout)": evaluate_and_store_metrics(MLPclassifierDropOut, features_test, labels_test, is_torch_model=True),
    }
    # Print results in tabular form
    print(f"\n{'Model':<40}{'Accuracy':<10}{'Precision':<10}{'Recall':<10}{'F1-Score':<10}")
    print("-" * 70)
    for model_name, metrics in results.items():
        print(f"{model_name:<40}{metrics['Accuracy']*100:<10.2f}{metrics['Precision']:<10.2f}{metrics['Recall']:<10.2f}{metrics['F1-Score']:<10.2f}")


