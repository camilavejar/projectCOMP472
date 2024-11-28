import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import dill

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        # Initialize the decision tree classifier
        # Max depth will be changed during hyperparameter tuning to see impact on accuracy
        self.max_depth = max_depth
        self.tree = None

    def buildTree(self, feature, label, depth=0):
        # Recursively build the tree
        # If the tree is a leaf node, return the most common class
        if len(np.unique(label)) == 1 or (self.max_depth and depth == self.max_depth):
            return {"class": np.unique(label)[0]}
        
        # Trying to split the data
        bestDataSplit = self.splitData(feature, label)
        # If we can't split the data, return the most common class
        if bestDataSplit is None:
            return {"class": np.bincount(label).argmax()}  
        # Else we continue to split the data
        left_tree = self.buildTree(feature[bestDataSplit['left_indices']], label[bestDataSplit['left_indices']], depth + 1)
        right_tree = self.buildTree(feature[bestDataSplit['right_indices']], label[bestDataSplit['right_indices']], depth + 1)
        
        return {
            "feature": bestDataSplit["feature"],
            "threshold": bestDataSplit["threshold"],
            "left": left_tree,
            "right": right_tree
            }

    def splitData(self, f, label):
        # Find the best split for the data using Gini impurity
        best_gini = float('inf')
        bestDataSplit = None

        for feature in range(f.shape[1]):
            thresholds = np.unique(f[:, feature])
            # Try all possible thresholds for the features and split into left and right
            for threshold in thresholds:
                left_indices = np.where(f[:, feature] <= threshold)[0]
                right_indices = np.where(f[:, feature] > threshold)[0]
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                # based on our gini index, we will find the best split
                gini = self.giniImpurity(label[left_indices], label[right_indices])
                if gini < best_gini:
                    best_gini = gini
                    bestDataSplit = {
                        "feature": feature,
                        "threshold": threshold,
                        "left_indices": left_indices,
                        "right_indices": right_indices
                    }

        return bestDataSplit

    def giniImpurity(self, left_y, right_y):
        # evaluates quality of a split 
        def gini(y):
            # Gini impurity = 1 - sum(p_i^2) for unique classes
            m = len(y)
            return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))
        # Calculate the weighted Gini impurity for both sides of the split
        left_gini = gini(left_y)
        right_gini = gini(right_y)
        m = len(left_y) + len(right_y)
        # Weighted Gini impurity = (size of left / total) * Gini(left) + (size of right / total) * Gini(right)
        weighted_gini = (len(left_y) / m) * left_gini + (len(right_y) / m) * right_gini
        # output determines which side of split is best
        return weighted_gini
    
    def fit(self, x, y):
        # Fit the decision tree classifier
        # Build the tree using the training data
        self.tree = self.buildTree(x, y)

    def predict(self, features):
        # Make predictions using the trained decision tree for all features
        return np.array([self.predictOneAtATime(f, self.tree) for f in features])

    def predictOneAtATime(self, f, tree):
        # Recursively predict the class for a single feature
        # If we reach a leaf node, return the class
        if "class" in tree:
            return tree["class"]
        # Else we continue to traverse the tree based on the feature and threshold
        if f[tree["feature"]] <= tree["threshold"]:
            return self.predictOneAtATime(f, tree["left"])
        else:
            return self.predictOneAtATime(f, tree["right"])

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

    ''' Python Decision Tree Classifier '''
    # Initialize the decision tree classifier with a max depth of 50
    model = DecisionTreeClassifier(max_depth=50)
    model.fit(features_train, labels_train)
    # Make predictions and calculate accuracy
    y_pred = model.predict(features_test)
    accuracy1 = np.mean(y_pred == labels_test)
    print('Python Decision Tree Classifier')
    print(f"Accuracy: {accuracy1 * 100:.2f}%")

    # Max depth  of 20
    model20 = DecisionTreeClassifier(max_depth=20)
    model20.fit(features_train, labels_train)
    y_pred20 = model20.predict(features_test)
    accuracy20 = np.mean(y_pred20 == labels_test)
    print('Python Decision Tree Classifier with max depth 20')
    print(f"Accuracy: {accuracy20 * 100:.2f}%")

    # Max depth of 80
    model80 = DecisionTreeClassifier(max_depth=80)
    model80.fit(features_train, labels_train)
    y_pred80 = model80.predict(features_test)
    accuracy80 = np.mean(y_pred80 == labels_test)
    print('Python Decision Tree Classifier with max depth 80')
    print(f"Accuracy: {accuracy80 * 100:.2f}%")

    ''' Scikit's Decision Tree Classifier '''
    # Initialize the model
    clf = DecisionTreeClassifier()
    # Fit the model
    clf.fit(features_train, labels_train)
    # Make predictions
    y_pred = clf.predict(features_test)
    # Evaluate accuracy
    accuracy = accuracy_score(labels_test, y_pred)
    print('Scikit\'s Decision Tree Classifier')
    print(f"Accuracy: {accuracy * 100:.2f}%")

    ### Save model
    # with open('decisionTreeClassifier50.dill', 'wb') as f:
    #     dill.dump(model, f)

    # with open('decisionTreeClassifier20.dill', 'wb') as f:
    #     dill.dump(model20, f)
    
    # with open('decisionTreeClassifier80.dill', 'wb') as f:
    #     dill.dump(model80, f)
    
    # with open('decisionTreeClassifierScikit.dill', 'wb') as f:
    #     dill.dump(clf, f)
