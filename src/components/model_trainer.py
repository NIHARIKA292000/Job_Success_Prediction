import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

from src.utils import save_object
from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainingConfig:
    trained_model_path: str = os.path.join("artifacts", "rf_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainingConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Starting Random Forest model training...")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Apply SMOTE
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logging.info(f"After SMOTE, training label distribution: {dict(pd.Series(y_train).value_counts())}")

            # Train model
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            try:
                auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            except Exception as e:
                logging.warning(f"AUC score could not be calculated: {e}")
                auc = 0.0

            logging.info(f"Model Performance - Accuracy: {accuracy:.2f}, F1 Score: {f1:.2f}, AUC: {auc:.2f}")

            if accuracy < 0.7 or f1 < 0.65 or auc < 0.7:
                raise CustomException(f"Model performance below threshold. Accuracy: {accuracy:.2f}, F1: {f1:.2f}, AUC: {auc:.2f}")

            # Save model
            save_object(
                file_path=self.config.trained_model_path,
                obj=model
            )

            logging.info("Model saved successfully.")
            return accuracy

        except Exception as e:
            raise CustomException(e, sys)
