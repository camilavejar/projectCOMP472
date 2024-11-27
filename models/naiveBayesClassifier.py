import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import pickle
import dill

class GaussianNaiveBayesClassifier:
    def __init__(self):
        # Initialize the model parameters
        self.classes = None  # Unique classes
        self.means = {}      # Mean of features for each class
        self.variances = {}  # Variance of features for each class
        self.priors = {}     # Prior probabilities of each class

    def fit(self, X, y):
        # Get all 10 classes with their labels y 
        self.classes = np.unique(y)
        for c in self.classes:
            X_c = X[y == c]  # Filter samples of class c
            self.means[c] = X_c.mean(axis=0)  # Feature means
            self.variances[c] = X_c.var(axis=0) + 1e-6  # Add epsilon to variances

            self.priors[c] = X_c.shape[0] / X.shape[0]  # Prior probability

    def likelihoodsWithGaussianPDF(self, x, mean, var):
        #Calculating likelihoods using gaussian probability density function
        # PDF: f(x) = (1 / sqrt(2 * pi * var)) * exp(-0.5 * ((x - mean) ** 2) / var)
        # Done one at a time for each feature
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * ((x - mean) ** 2) / var)
 

    def _class_likelihood(self, X, c):
        # Calculate likelihood of each feature for class c with Gaussian PDF
        mean = self.means[c]
        var = self.variances[c]
        likelihoods = self.likelihoodsWithGaussianPDF(X, mean, var)
        return np.clip(likelihoods, 1e-2, None)  # Clip to avoid zeros, else get zero log errors

    def predict(self, X):
        # Used to predict the class of the input feature vector for Evaluation
        posteriors = []  # List to hold log-posterior probabilities for each class
        for c in self.classes:
            # Using sum of log-prior and log-likelihood to avoid small numbers
            prior = np.log(self.priors[c])  
            likelihood = np.log(self._class_likelihood(X, c))  
            posterior = prior + np.sum(likelihood, axis=1)  
            posteriors.append(posterior)
        # Output the class with the highest posterior probability
        posteriors = np.array(posteriors)  
        return self.classes[np.argmax(posteriors, axis=0)]  
    
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

    ''' Python Naive Bayes Classifier '''
    # Initialize and train the Naive Bayes model
    model = GaussianNaiveBayesClassifier()
    # Fit the model
    model.fit(features_train, labels_train)
    # Make predictions
    y_pred = model.predict(features_test)
    # Evaluate accuracy
    accuracy1 = np.mean(y_pred == labels_test)
    print('Python Naive Bayes Classifier')
    print(f"Accuracy: {accuracy1 * 100:.2f}%")

    ### Save model
    with open('gaussianNaiveBayesClassifier.dill', 'wb') as f:
        dill.dump(model, f)

    ''' Scikit's Naive Bayes Classifier '''
    # Initialize and train the model
    gnb = GaussianNB()
    gnb.fit(features_train, labels_train)
    # Evaluate the model
    accuracy = gnb.score(features_test, labels_test)
    print('Scikit\'s Naive Bayes Classifier')
    print(f"Accuracy: {accuracy * 100:.2f}%")