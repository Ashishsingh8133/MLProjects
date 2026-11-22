## This file contains machine larning  model training
import os 
import sys 
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from sklearn.ensemble import (
    GradientBoostingRegressor,
    AdaBoostRegressor,RandomForestRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score


from src.utils import save_object,evaluate_model




ROOT_DIR=os.path.abspath(os.path.join(os.path.dirname(__file__),'../../'))

@dataclass
class ModelTrainerConfig:
    Trained_model_file_path=os.path.join(ROOT_DIR,'artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_training_config=ModelTrainerConfig()

    def initiate_model_training(self,train_arr,test_arr):
        logging.info("initiating model training")
        try:
            logging.info("Splitting train & test data")
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models={
                'GradientBoost':GradientBoostingRegressor(),
                'Adaboost':AdaBoostRegressor(),
                'RandomForest':RandomForestRegressor(),
                'linear_Regression':LinearRegression(),
                'Support_vector_regressor':SVR(),
                'DecisionTreeRegressor':DecisionTreeRegressor(),
                'Neighbors':KNeighborsRegressor(),
            }

            param={
                'GradientBoost':{
                    'learning_rate':[0.1,0.01,0.5],
                    'n_estimators':[8,16,32,64,128]
                },
                'linear_Regression':{

                },
                'Adaboost':{
                    'n_estimators':[8,16,32,64,128],
                    'learning_rate':[0.5,0.01,0.1]
                },
                'RandomForest':{
                    'n_estimators':[8,16,32,64,128]
                },
                'Support_vector_regressor':{
                    'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
                    'degree':[3,5,4,6],
                    'gamma':['scale', 'auto']
                },
                'DecisionTreeRegressor':{       
                    'max_depth':[2,3,4,5,7],
                    'max_features':['sqrt', 'log2']
                },
                'Neighbors':{
                    'n_neighbors':[5,6,7,8]
                }
            }

            model_report:dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,params=param)
            logging.info("Training the model over training data")
            logging.info("Testing the model over testing data")

            ##to get my best model score from dictionary
            best_model_score=max(sorted(model_report.values()))

            #to get best model_name from dictionary
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name] ##pick the model from models dictionary
            logging.info(f"the best model is {best_model}")

            if best_model_score<0.6:
                raise CustomException("No model is performing best over this dataset")
            

            save_object(
                file_path=self.model_training_config.Trained_model_file_path,
                obj=best_model
            )

            Predicted=best_model.predict(x_test)
            r2_square=r2_score(y_test,Predicted)
            
            
            return r2_square
        except Exception as e:
            raise CustomException(e,sys)





