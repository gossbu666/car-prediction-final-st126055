import numpy as np
import math
import matplotlib.pyplot as plt

class LogisticRegression:
    """
    A from-scratch implementation of Multinomial Logistic Regression.
    
    Features:
    - L2 (Ridge) Regularization
    - Batch, Mini-Batch, and Stochastic Gradient Descent
    - Loss history tracking for diagnostics
    """
    def __init__(self, learning_rate=0.1, n_iterations=100, lmbda=0.01, batch_size=32):
        """
        Initializes the model.
        
        Args:
            learning_rate (float): Step size for gradient descent.
            n_iterations (int): Number of epochs (full passes over the training data).
            lmbda (float): Regularization strength (lambda).
            batch_size (int): Number of samples per batch.
                - If batch_size == n_samples -> Batch Gradient Descent
                - If batch_size == 1 -> Stochastic Gradient Descent (SGD)
                - Otherwise -> Mini-Batch Gradient Descent
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lmbda = lmbda
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _one_hot(self, y, n_classes):
        """Helper function to convert labels to one-hot encoding."""
        ohe = np.zeros((len(y), n_classes))
        ohe[np.arange(len(y)), y] = 1
        return ohe

    def softmax(self, z):
        """The softmax activation function for converting scores to probabilities."""
        # Subtracting max(z) is a trick for numerical stability to prevent overflow
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        """Trains the model using the provided data."""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Initialize weights and bias to zeros
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)
        
        # The main loop now iterates over epochs
        for epoch in range(self.n_iterations):
            # --- CRITICAL STEP: Shuffle the data at the start of each epoch ---
            permutation = np.random.permutation(n_samples)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]
            
            # --- Loop over mini-batches ---
            n_batches = math.ceil(n_samples / self.batch_size)
            for i in range(n_batches):
                # Get the current mini-batch
                start = i * self.batch_size
                end = min(start + self.batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                # --- The logic below uses the BATCH data for one update ---
                n_batch_samples = X_batch.shape[0]
                if n_batch_samples == 0: continue
                
                y_ohe_batch = self._one_hot(y_batch, n_classes)

                # 1. Calculate scores (logits) and probabilities
                z = X_batch @ self.weights + self.bias
                y_hat = self.softmax(z)
                
                # 2. Calculate gradient with L2 penalty
                original_gradient = (1 / n_batch_samples) * (X_batch.T @ (y_hat - y_ohe_batch))
                l2_penalty = (self.lmbda / n_batch_samples) * self.weights
                gradient = original_gradient + l2_penalty
                
                # 3. Update weights
                self.weights -= self.learning_rate * gradient

            # --- Calculate and store loss on the FULL dataset at the end of each epoch ---
            full_z = X @ self.weights + self.bias
            full_y_hat = self.softmax(full_z)
            full_y_ohe = self._one_hot(y, n_classes)
            
            cross_entropy_loss = -np.mean(np.sum(full_y_ohe * np.log(full_y_hat + 1e-8), axis=1))
            l2_loss_penalty = (self.lmbda / (2 * n_samples)) * np.sum(self.weights**2)
            total_loss = cross_entropy_loss + l2_loss_penalty
            self.loss_history.append(total_loss)
            
            if (epoch % 10 == 0) or (epoch == self.n_iterations - 1):
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    def predict(self, X):
        """Makes predictions for new data."""
        z = X @ self.weights + self.bias
        y_hat = self.softmax(z)
        return np.argmax(y_hat, axis=1)