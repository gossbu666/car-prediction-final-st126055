import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class LogisticRegression(BaseEstimator, ClassifierMixin):
    """
    A robust, from-scratch implementation of Multinomial Logistic Regression.
    Includes L2 regularization, mini-batch gradient descent, and input validation.
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

    # ---------- internal helpers ----------

    def _one_hot(self, y, n_classes):
        ohe = np.zeros((len(y), n_classes))
        ohe[np.arange(len(y)), y] = 1
        return ohe

    def _softmax(self, z):
        # numerical stability
        z_safe = np.clip(z, -250, 250)
        exp_z = np.exp(z_safe - np.max(z_safe, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # ---------- training ----------
    def fit(self, X, y):
        # --- validation ---
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array for X, got {X.ndim}D.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Mismatched samples: X has {X.shape[0]}, y has {y.shape[0]}.")

        n_samples, n_features = X.shape
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)

        # store label mapping
        self.class_mapping = {label: i for i, label in enumerate(unique_classes)}
        y_mapped = np.array([self.class_mapping[label] for label in y])

        # initialize params
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)
        epsilon = 1e-15

        # --- training loop ---
        for epoch in range(self.n_iterations):
            permutation = np.random.permutation(n_samples)
            X_shuffled = X[permutation]
            y_shuffled = y_mapped[permutation]

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i : i + self.batch_size]
                y_batch = y_shuffled[i : i + self.batch_size]
                n_batch_samples = X_batch.shape[0]

                y_ohe_batch = self._one_hot(y_batch, n_classes)

                z = X_batch @ self.weights + self.bias
                y_hat = self._softmax(z)

                dw = (
                    (1 / n_batch_samples) * (X_batch.T @ (y_hat - y_ohe_batch))
                    + (self.lmbda / n_batch_samples) * self.weights
                )
                db = (1 / n_batch_samples) * np.sum(y_hat - y_ohe_batch, axis=0)

                # gradient clipping
                grad_norm = np.linalg.norm(dw)
                clip_threshold = 5.0
                if grad_norm > clip_threshold:
                    dw = dw * clip_threshold / grad_norm

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            # --- epoch loss ---
            full_z = X @ self.weights + self.bias
            full_y_hat = self._softmax(full_z)
            full_y_ohe = self._one_hot(y_mapped, n_classes)

            clipped_y_hat = np.clip(full_y_hat, epsilon, 1 - epsilon)
            cross_entropy_loss = -np.mean(
                np.sum(full_y_ohe * np.log(clipped_y_hat), axis=1)
            )
            l2_penalty = (self.lmbda / (2 * n_samples)) * np.sum(self.weights**2)
            total_loss = cross_entropy_loss + l2_penalty

            if np.isnan(total_loss) or np.isinf(total_loss):
                break
            self.loss_history.append(total_loss)

        return self

    # ---------- prediction ----------
    def predict_proba(self, X):
        # --- validation ---
        if self.weights is None or self.bias is None:
            raise ValueError("Model is not fitted yet. Call fit(X, y) before predict.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array for X, got {X.ndim}D.")

        n_features_model = self.weights.shape[0]
        if X.shape[1] != n_features_model:
            raise ValueError(
                f"Expected X with {n_features_model} features, got {X.shape[1]}."
            )

        z = X @ self.weights + self.bias
        return self._softmax(z)

    def predict(self, X):
        P = self.predict_proba(X)
        predicted_indices = np.argmax(P, axis=1)

        # reverse mapping to original labels
        reverse_class_mapping = {i: label for label, i in self.class_mapping.items()}
        return np.array([reverse_class_mapping[i] for i in predicted_indices])