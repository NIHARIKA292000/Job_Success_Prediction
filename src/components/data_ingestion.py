import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion process...")

        try:
            # Read the original dataset
            df = pd.read_csv("notebook/data/HR_comma_sep.csv")
            logging.info("Successfully read the dataset into a DataFrame.")

            # Create directories if they don't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved to artifacts/raw.csv")

            # Train-test split
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Train-test split completed.")

            # Save train and test sets
            train_df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train and test datasets saved to artifacts folder.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            logging.error("Exception occurred during data ingestion")
            raise CustomException(e, sys)

# Optional test run
if __name__ == "__main__":
    ingestion = DataIngestion
