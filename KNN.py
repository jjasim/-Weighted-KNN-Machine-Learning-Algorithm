import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(x1-x2)**2)

class KNN:
    def __init__(self, k=3):
        self.k = k

    # Training method
    # IMPORTANT: X and y have to be numpy arrays!
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    # Run the predictions on the test samples
    # Note: The capital X here denotes that multiple sample values of X can be expected
    # Eg. color, weight, height etc
    # This function returns predicted values as a numpy array
    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
    
    # This helper function predicts for one sample
    # Hence the small x
    def _predict(self, x):
        # Compute distances using Euclidean distances 
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Get K-nearest neighbours for a particular sample
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Get the most common sample-label in the K-nearest neighbours
        most_common = Counter(k_nearest_labels).most_common(1)

        return most_common[0][0]

    # Returns the accuracy of the predictions compared with the actual values in test values
    def accuracy(self, predictions, y_test):
        return np.sum(predictions == y_test) / len(y_test)

