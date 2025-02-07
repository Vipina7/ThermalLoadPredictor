import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import dill
from tqdm import tqdm

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Saved preprocessing object")

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, X_test, y_train, y_test, models, param):
    try:
        test_report = {}

        logging.info('Model training initiated')
        for i in tqdm(range(len(list(models)))):
            model = list(models.values())[i]
            params=param[list(models.keys())[i]]

            gs = GridSearchCV(model,params,cv=3, n_jobs=-1)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = []
            test_model_score = []

            # train model scores
            rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            mae = mean_absolute_error(y_train, y_train_pred)
            r2 = r2_score(y_train, y_train_pred)

            train_model_score.extend([rmse, mae, r2])

            #test model scores
            rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
            mae_test = mean_absolute_error(y_test, y_test_pred)
            r2_test = r2_score(y_test, y_test_pred)
            
            test_model_score.extend([rmse_test, mae_test, r2_test])

            test_report[list(models.keys())[i]] = test_model_score
                
            logging.info('Model performance reports generated')

            test_report_df = pd.DataFrame.from_dict(test_report, orient = 'index', columns = ["RMSE", "MAE", "R²"])
            
        test_report_df.to_csv('artifacts/model_performance.csv')
        return test_report_df
         
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)