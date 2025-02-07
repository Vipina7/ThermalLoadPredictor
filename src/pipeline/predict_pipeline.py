import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            logging.info("Loading the model and preprocessor object")
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            logging.info("Preprocessing the input features")
            scale_cols = ['surface_area_m2', 'wall_area_m2', 'overall_height_m', 'glazing_area_m2']
            features[scale_cols] = preprocessor.transform(features[scale_cols])
            
            logging.info("Making predictions successful")
            prediction = model.predict(features)
            return prediction
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self, 
                 surface_area_m2, 
                 wall_area_m2, 
                 overall_height_m, 
                 glazing_area_m2, 
                 glazing_area_Distribution):
        
        self.surface_area_m2 = surface_area_m2
        self.wall_area_m2 = wall_area_m2
        self.overall_height_m = overall_height_m
        self.glazing_area_m2 = glazing_area_m2
        self.glazing_area_Distribution = glazing_area_Distribution


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "surface_area_m2": [self.surface_area_m2],
                "wall_area_m2": [self.wall_area_m2],
                "overall_height_m": [self.overall_height_m],
                "glazing_area_m2": [self.glazing_area_m2],
                "glazing_area_Distribution": [self.glazing_area_Distribution]
            }
            logging.info("Dataframe ready for prediction")

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)
