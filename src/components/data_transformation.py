import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataTransformationConfig:
    transformation_obj_path:str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_preprocessor_object(self):
        try:
            logging.info("Data preprocessing initiated")

            scaler = StandardScaler()
            logging.info('Preprocessor object obtained')

            return scaler
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Importing train and test sets")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Obtaining preprocessor object')
            preprocessor_obj = self.get_preprocessor_object()

            target = ['heating_load(kWh/m²)','cooling_load(kWh/m²)']

            X_train = train_df.drop(columns=target , axis=1)
            y_train = train_df[target]

            X_test = test_df.drop(columns=target, axis=1)
            y_test = test_df[target]
            logging.info("Obtained independent and dependent features")
            
            scale_cols = ['surface_area_m2', 'wall_area_m2', 'overall_height_m', 'glazing_area_m2']

            X_train[scale_cols] = preprocessor_obj.fit_transform(X_train[scale_cols])
            X_test[scale_cols] = preprocessor_obj.transform(X_test[scale_cols])
            logging.info("Data transformation complete")

            save_object(
                file_path=self.data_transformation_config.transformation_obj_path,
                obj=preprocessor_obj)
            
            return (
                X_train,
                X_test,
                y_train,
                y_test
            )
        
        except Exception as e:
            raise CustomException(e,sys)