import os
import sys

from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components import data_transformation
from src.components import model_trainer

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', "train.csv")
    test_data_path: str=os.path.join('artifacts', "test.csv")
    raw_data_path: str=os.path.join('artifacts', "data.csv")



class DataIngestion:
    def __init__(self):
        self.ingestion_congif = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook\data\insurance.csv')
            logging.info('Read the dataset as dataframe')
            
            # create the artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_congif.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_congif.raw_data_path,index=False, header=True)
            
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df,test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_congif.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_congif.test_data_path, index=False, header=True)
            logging.info("Ingestion of the data is completed")

            return (self.ingestion_congif.train_data_path, self.ingestion_congif.test_data_path)

        except Exception as e:
            raise CustomException(e, sys)
        


if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()


    data_transform_obj = data_transformation.DataTransformation()
    tr_ar, test_arr, pre_path = data_transform_obj.initiate_data_transformation(train_data_path, test_data_path)

    model_trainer_obj = model_trainer.ModelTrainer()
    print(model_trainer_obj.initiate_model_trainer(tr_ar, test_arr, pre_path))


