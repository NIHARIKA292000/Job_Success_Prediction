import os
import sys
import pandas as pd
import numpy as np

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

def main():
    try:
        logging.info("🚀 Starting the model training pipeline...")

        # Step 1: Data Ingestion
        ingestion = DataIngestion()
        train_data_path, test_data_path = ingestion.initiate_data_ingestion()
        logging.info(f"✅ Data ingestion completed. Train: {train_data_path}, Test: {test_data_path}")

        # Step 2: Data Transformation
        transformation = DataTransformation()
        train_array, test_array, _ = transformation.initiate_data_transformation(train_data_path, test_data_path)
        logging.info("✅ Data transformation completed.")

        # Step 3: Model Training
        trainer = ModelTrainer()
        final_accuracy = trainer.initiate_model_trainer(train_array, test_array)
        logging.info(f"✅ Model training completed. Final Test Accuracy: {final_accuracy:.4f}")

        print(f"\n🎯 Final Random Forest Test Accuracy: {final_accuracy:.4f}")

    except Exception as e:
        logging.error("❌ Training pipeline failed due to an exception.")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()
