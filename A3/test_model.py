import pytest
import numpy as np
# Make sure your LogisticRegression class is in a file named model.py
# If it's in your notebook, you'll need to copy the class into a .py file.
from model import LogisticRegression 

# --- ARRANGE: A fixture to create and train a model once for all tests ---
# A "fixture" is a setup function that pytest runs before your tests.
@pytest.fixture
def trained_model():
    """
    Creates and trains a LogisticRegression model on simple mock data.
    This trained model is then available to all test functions that need it.
    """
    # Arrange: Create simple mock data
    X_train = np.random.rand(100, 10) # 100 samples, 10 features
    y_train = np.random.randint(0, 4, 100) # 4 classes
    
    # Act: Train the model
    model = LogisticRegression(n_iterations=10) # Train for a few iterations
    model.fit(X_train, y_train)
    
    return model

# --- Test 1: Does the model handle the expected input shape correctly? ---
def test_model_input_shape(trained_model):
    """
    Tests if the model's predict function runs without errors on an
    input with the correct shape.pytest
    """
    # Arrange: Create mock data with the same number of features (10)
    # The number of samples (5) can be different.
    n_features = 10
    X_test = np.random.rand(5, n_features)
    
    try:
        # Act: Make a prediction
        trained_model.predict(X_test)
    except Exception as e:
        # Assert: If any exception occurs, the test fails.
        pytest.fail(f"Model prediction failed with valid input shape. Error: {e}")

# --- Test 2: Does the model produce the expected output shape? ---
def test_model_output_shape(trained_model):
    """
    Tests if the model's output has the correct shape, which should be
    a 1D array with a length equal to the number of input samples.
    """
    # Arrange: Create mock test data
    n_samples = 15
    n_features = 10
    X_test = np.random.rand(n_samples, n_features)
    
    # Act: Make a prediction
    predictions = trained_model.predict(X_test)
    
    # Assert: Check if the output shape is correct
    assert predictions.shape == (n_samples,), f"Output shape was {predictions.shape}, expected {(n_samples,)}"