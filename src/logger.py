import logging 
import os 
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%d_%m_%y_%H_%M_%S')}.log" ##create log file
logs_path=os.path.join(os.getcwd(),"log") ##this will create a path to log file
os.makedirs(logs_path,exist_ok=True) ##create directory by provided log path 

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE) #file path

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s-%(message)s',
    level=logging.INFO,
    
)

if __name__=="__main__":
    logging.info("Logging has started")
