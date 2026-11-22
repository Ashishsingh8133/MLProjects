##After ingesting the data the next step is to transform that data so  next component is data_transform
##this file contains all the code related to data transformation like Onehot encoding,Label Encoding,Standardization
import os
from dataclasses import dataclass
import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.utils import save_object
from src.exception import CustomException
from src.logger import logging

ROOT_DIR=os.path.abspath(os.path.join(os.path.dirname(__file__),"../../"))

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file=os.path.join(ROOT_DIR,'artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        logging.info("Data Transformation has started")
        try:
            numerical_columns=[ 'reading score', 'writing score']
            categorical_columns=['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("Scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("OneHotEncode",OneHotEncoder()),
                   
                ]
            )
            logging.info(f"Categorical Columns:{categorical_columns}")
            logging.info(f"Numerical Columns:{numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("num pipelinee",num_pipeline,numerical_columns),
                    ("cat pipeline",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor


        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)
            logging.info("Reading Train and Test Data Completed")

            logging.info("Obtaining preprocessing object")

            preprocessor_obj=self.get_data_transformer_object()

            target_column_name="math score"
            numerical_columns=[ 'reading score', 'writing score']

            #dependent and independent feature for training data
            input_feature_train_df=train_df.drop(columns=[target_column_name])
            target_feature_train_df=train_df[target_column_name]

            ##dependent and independent feature for testing data
            input_feature_test_df=test_df.drop(columns=[target_column_name])
            target_feature_test_df=test_df[target_column_name]

            logging.info(" Applying preprocessing object on the training dataframe and testing dataframe")

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            #concatenate array column wise using np.c_
            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]

            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info(f"Saving Preprocessing object:")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file,
                obj=preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file
            )
        except Exception as e:
            raise CustomException(e,sys)










