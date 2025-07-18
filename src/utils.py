import os
import sys
import dill
import numpy as np
from sklearn.metrics import accuracy_score
from src.exception import CustomException

def save_object(file_path, obj):
    """
    Save any Python object to disk using dill.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Load a Python object from disk using dill.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, model):
    """
    Train the provided model, evaluate it, and return accuracy metrics.
    """
    try:
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        return {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy
        }

    except Exception as e:
        raise CustomException(e, sys)
