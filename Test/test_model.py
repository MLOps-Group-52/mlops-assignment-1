import pytest
import joblib
import os
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_and_evaluate_model(model_filename='model.joblib'):
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Decision Tree model
    model = DecisionTreeClassifier()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    # Serialize the model to a file
    joblib.dump(model, model_filename)

    return accuracy, model_filename, X_test, y_test, y_pred

def test_model_accuracy():
    accuracy, _, _, _, _ = train_and_evaluate_model()
    assert accuracy > 0.9, "Model accuracy is below threshold"

def test_model_file():
    _, model_filename, _, _, _ = train_and_evaluate_model()
    assert os.path.exists(model_filename), "Model file was not created"

    # Check if the model can be loaded
    loaded_model = joblib.load(model_filename)
    assert isinstance(loaded_model, DecisionTreeClassifier), "Loaded model is not of type DecisionTreeClassifier"

    # Clean up model file after test
    os.remove(model_filename)

def test_model_predictions():
    accuracy, _, X_test, y_test, y_pred = train_and_evaluate_model()

    # Verify that predictions are consistent with accuracy
    loaded_model = joblib.load('model.joblib')
    y_pred_loaded = loaded_model.predict(X_test)

    assert (y_pred == y_pred_loaded).all(), "Model predictions do not match after loading"

    # Clean up model file after test
    os.remove('model.joblib')