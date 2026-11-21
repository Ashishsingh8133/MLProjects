##data ingestion is nothing but a part of module that module is a component , this data ingestion is used to read the data from various sources
##this file will have the code that is something related to reading the data from any source 
import os 
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


##we create another class to save the train_data, test_data and some other raw data that class called as data_ingestion class
@dataclass #with the help of this decorator i can directly define the variables in the class DataIngestionConfig , we don't need init to define variables
class DataIngestionConfig:
    train_data_path:str=os.path.join(ROOT_DIR,"artifacts","Train.csv") #save the train data at the given path
    test_data_path:str=os.path.join(ROOT_DIR,"artifacts","Test.csv") #save test data at the given path
    raw_data_path:str=os.path.join(ROOT_DIR,"artifacts","Data.csv") #save the raw data at the given path



class DataIngestion:
    def __init__(self):
        self.IngestionConfig=DataIngestionConfig() #here when we call DataIngestion then all three paths will be saved in IngestionConfig
      
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method and component")
        try:
            df = pd.read_csv("/Users/ashish/ML Projects/notebook/data/stud.csv") #reading the dataset
            logging.info("Read the dataset as DataFrame")
            
            os.makedirs(os.path.dirname(self.IngestionConfig.train_data_path),exist_ok=True) #save train data into the directory
            df.to_csv(self.IngestionConfig.raw_data_path,index=False,header=True) # saving the raw data into .csv file

            logging.info("Train Test Split Initiated")
            train_set,test_set=train_test_split(df,test_size=0.20,random_state=42)
            train_set.to_csv(self.IngestionConfig.train_data_path,index=False,header=True)
            test_set.to_csv(self.IngestionConfig.test_data_path,index=False,header=True)

            logging.info("ingestion of the data has been completed")
            return(
                self.IngestionConfig.train_data_path,
                self.IngestionConfig.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()
        
            
    

