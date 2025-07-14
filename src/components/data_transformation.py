import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        Creates and returns the preprocessing pipeline
        '''
        try:
            numerical_columns = ['satisfaction_level', 'last_evaluation', 'number_project', 
                                 'average_montly_hours', 'time_spend_company']
            categorical_columns = ['Department', 'salary']

            # Pipeline for numerical features
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # Pipeline for categorical features
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot", OneHotEncoder(drop='first', handle_unknown='ignore'))
            ])

            # Combine into column transformer
            preprocessor = ColumnTransformer(transformers=[
                ("num", num_pipeline, numerical_columns),
                ("cat", cat_pipeline, categorical_columns)
            ])

            logging.info("Preprocessing pipeline created successfully.")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded.")

            # Define features and target
            target_column = "left"

            input_features_train = train_df.drop(columns=[target_column])
            target_feature_train = train_df[target_column]

            input_features_test = test_df.drop(columns=[target_column])
            target_feature_test = test_df[target_column]

            # Get preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            # Fit and transform
            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train)
            input_feature_test_arr = preprocessing_obj.transform(input_features_test)

            # Combine input and target arrays
            train_arr = np.c_[input_feature_train_arr, target_feature_train.to_numpy()]
            test_arr = np.c_[input_feature_test_arr, target_feature_test.to_numpy()]

            # Save preprocessor
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Data transformation complete. Preprocessor saved.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
