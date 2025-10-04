import numpy as np
import math
from sklearn.base import BaseEstimator, ClassifierMixin

class LogisticRegression(BaseEstimator, ClassifierMixin):
    """
    A robust, from-scratch implementation of Multinomial Logistic Regression.
    """
    def __init__(self, learning_rate=0.1, n_iterations=100, lmbda=0.01, batch_size=32):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lmbda = lmbda
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.class_mapping = {}

    def _one_hot(self, y, n_classes):
        ohe = np.zeros((len(y), n_classes))
        ohe[np.arange(len(y)), y] = 1
        return ohe

    def _softmax(self, z):
        z_safe = np.clip(z, -250, 250)
        exp_z = np.exp(z_safe - np.max(z_safe, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        
        # Store the mapping from original class labels to integer indices
        self.class_mapping = {label: i for i, label in enumerate(unique_classes)}
        y_mapped = np.array([self.class_mapping[label] for label in y])

        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)
        epsilon = 1e-15

        for epoch in range(self.n_iterations):
            permutation = np.random.permutation(n_samples)
            X_shuffled = X[permutation]
            y_shuffled = y_mapped[permutation]
            
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                n_batch_samples = X_batch.shape[0]

                y_ohe_batch = self._one_hot(y_batch, n_classes)

                z = X_batch @ self.weights + self.bias
                y_hat = self._softmax(z)
                
                dw = (1/n_batch_samples) * (X_batch.T @ (y_hat - y_ohe_batch)) + (self.lmbda / n_batch_samples) * self.weights
                db = (1/n_batch_samples) * np.sum(y_hat - y_ohe_batch, axis=0)
                
                grad_norm = np.linalg.norm(dw)
                clip_threshold = 5.0
                if grad_norm > clip_threshold:
                    dw = dw * clip_threshold / grad_norm

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            full_z = X @ self.weights + self.bias
            full_y_hat = self._softmax(full_z)
            full_y_ohe = self._one_hot(y_mapped, n_classes)
            
            clipped_y_hat = np.clip(full_y_hat, epsilon, 1 - epsilon)
            cross_entropy_loss = -np.mean(np.sum(full_y_ohe * np.log(clipped_y_hat), axis=1))
            l2_penalty = (self.lmbda / (2 * n_samples)) * np.sum(self.weights**2)
            total_loss = cross_entropy_loss + l2_penalty
            
            if np.isnan(total_loss) or np.isinf(total_loss):
                break
            self.loss_history.append(total_loss)
        return self
            
    def predict(self, X):
        z = X @ self.weights + self.bias
        y_hat = self._softmax(z)
        predicted_indices = np.argmax(y_hat, axis=1)
        
        # Create a reverse mapping to get original class labels back
        reverse_class_mapping = {i: label for label, i in self.class_mapping.items()}
        return np.array([reverse_class_mapping[i] for i in predicted_indices])