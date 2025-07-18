import os
import sys
import pickle
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("rf_model.pkl")
    feature_columns_file_path: str = os.path.join("feature_columns.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("🔧 Splitting training and test data...")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            logging.info("📦 Training RandomForest model...")
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            logging.info("🔍 Evaluating model...")
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            logging.info(f"✅ Model accuracy on test set: {accuracy:.4f}")

            # Save trained model
            with open(self.config.trained_model_file_path, "wb") as f:
                pickle.dump(model, f)
                logging.info(f"💾 Model saved to {self.config.trained_model_file_path}")

            # Save feature columns (for prediction-time use in app.py)
            feature_columns = [f"feature_{i}" for i in range(X_train.shape[1])]
            with open(self.config.feature_columns_file_path, "wb") as f:
                pickle.dump(feature_columns, f)
                logging.info(f"📁 Feature columns saved to {self.config.feature_columns_file_path}")

            return accuracy

        except Exception as e:
            logging.error("❌ Exception occurred in ModelTrainer")
            raise CustomException(e, sys)

