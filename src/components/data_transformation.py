import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj




@dataclass
class DataTransformaitonConfig:
    preprocessor_ob_file_path = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformaitonConfig()

    def get_data_transformer_obj(self):

        # data transformation is performed in this function

        try:
            num_columns = ['age','bmi','children']
            cat_columns = ['sex','smoker','region']

            num_pipeline = Pipeline(
                steps=[('scaling',StandardScaler())]
            )

            cat_pipeline = Pipeline(
                steps=[('oh_encode',OneHotEncoder()), ('scaling',StandardScaler(with_mean=False))]
            )
            
            # column transformation object created

            preprocessor = ColumnTransformer(
                [('num_pipeline', num_pipeline, num_columns), ('cat_pipeline', cat_pipeline, cat_columns)]
            )

            logging.info('feature transformation object is created')

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('train and test df has been loaded')

            preprocessor_obj = self.get_data_transformer_obj()

            target_column = 'charges'
            #num_columns = ['age','bmi','children']
            
            # create INPUT and TARGET for train and test data
            logging.info('input and target data are being created..')
            input_feat_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feat_train_df = train_df[target_column]

            input_feat_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feat_test_df = test_df[target_column]

            logging.info('input and target data creation is completed')

            logging.info('Preprocessing is being applied on input features for transformation..')

            input_feat_train_arr = preprocessor_obj.fit_transform(input_feat_train_df)
            input_feat_test_arr = preprocessor_obj.transform(input_feat_test_df)


            train_arr = np.c_[input_feat_train_arr, np.array(target_feat_train_df)]
            test_arr = np.c_[input_feat_test_arr, np.array(target_feat_test_df)]


            logging.info('saved preprocessing object')

            save_obj(file_path = self.data_transformation_config.preprocessor_ob_file_path,
                        obj = preprocessor_obj)
            

            return train_arr, test_arr, self.data_transformation_config.preprocessor_ob_file_path


        except Exception as e:
            raise CustomException(e,sys)
            



