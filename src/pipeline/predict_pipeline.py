import os
import sys
import pandas as pd

# Ensure that we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exception import CustomException
from utils import load_object

class PredictPipeline:
    def __init__(self):
        # Go up two directories to reach root where 'artifacts' folder exists
        self.model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts', 'rf_model.pkl')
        self.preprocessor_path = os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts', 'preprocessor.pkl')

        # Normalize paths
        self.model_path = os.path.abspath(self.model_path)
        self.preprocessor_path = os.path.abspath(self.preprocessor_path)

    def predict(self, features: pd.DataFrame):
        try:
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)
            return prediction

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 satisfaction_level: float,
                 last_evaluation: float,
                 number_project: int,
                 average_montly_hours: int,
                 time_spend_company: int,
                 Work_accident: int,
                 promotion_last_5years: int,
                 Department: str,
                 salary: str):
        self.satisfaction_level = satisfaction_level
        self.last_evaluation = last_evaluation
        self.number_project = number_project
        self.average_montly_hours = average_montly_hours
        self.time_spend_company = time_spend_company
        self.Work_accident = Work_accident
        self.promotion_last_5years = promotion_last_5years
        self.Department = Department
        self.salary = salary

    def get_data_as_data_frame(self):
        try:
            data = {
                "satisfaction_level": [self.satisfaction_level],
                "last_evaluation": [self.last_evaluation],
                "number_project": [self.number_project],
                "average_montly_hours": [self.average_montly_hours],
                "time_spend_company": [self.time_spend_company],
                "Work_accident": [self.Work_accident],
                "promotion_last_5years": [self.promotion_last_5years],
                "Department": [self.Department],
                "salary": [self.salary],
            }
            return pd.DataFrame(data)

        except Exception as e:
            raise CustomException(e, sys)
