import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object
from dataclasses import dataclass

from tqdm import tqdm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score

class ModelTrainerConfig:
    trained_model_path:str = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, X_train, X_test, y_train, y_test):
        try:
            models = {
                "Decision Tree": MultiOutputRegressor(DecisionTreeRegressor()),
                "Random Forest": MultiOutputRegressor(RandomForestRegressor()),
                "Gradient Boosting": MultiOutputRegressor(GradientBoostingRegressor()),
                "Linear Regression": MultiOutputRegressor(LinearRegression()),
                "XGBRegressor": MultiOutputRegressor(XGBRegressor(objective='reg:squarederror')),
                "CatBoost Regressor": MultiOutputRegressor(CatBoostRegressor(verbose=False)),
                "AdaBoost Regressor": MultiOutputRegressor(AdaBoostRegressor()),
                }

            
            # Hyperparameter Grid (for Multi-Output Models)
            params = {
                "Decision Tree": {
                'estimator__criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                'estimator__splitter': ['best', 'random'],
                'estimator__max_features': ['sqrt', 'log2'],
                },
                "Random Forest": {
                    'estimator__criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'estimator__max_features': ['sqrt', 'log2', None],
                    'estimator__n_estimators': [32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'estimator__loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'estimator__learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'estimator__subsample': [0.7, 0.75, 0.8, 0.85, 0.9],
                    'estimator__criterion': ['squared_error', 'friedman_mse'],
                    'estimator__max_features': [1.0, 'sqrt', 'log2'],
                    'estimator__n_estimators': [32, 64, 128, 256]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'estimator__learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'estimator__n_estimators': [32, 64, 128, 256]
                },
                "CatBoost Regressor": {
                    'estimator__depth': [6, 8, 10],
                    'estimator__learning_rate': [0.01, 0.05, 0.1],
                    'estimator__iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'estimator__learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'estimator__loss': ['linear', 'square', 'exponential'],
                    'estimator__n_estimators': [32, 64, 128, 256]
                }
            }
            
            report = evaluate_models(X_train, X_test, y_train, y_test, models, params)

            best_model_score = report['R²'].max()
            best_model_name = report[report['R²']== best_model_score].index[0]
            best_model =models[best_model_name]
            
            save_object(
                file_path = self.model_trainer_config.trained_model_path,
                obj = best_model
            )
            logging.info('Saved the model')

            predicted=best_model.predict(X_test)

            return r2_score(y_test, predicted)
        
        except Exception as e:
            raise CustomException(e, sys)