import sys
from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:

    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("Training pipeline started")

            # Step 1: Data Ingestion
            data_ingestion = DataIngestion()
            train_path, test_path = data_ingestion.initiate_data_ingestion()

            # Step 2: Data Transformation
            data_transformation = DataTransformation()
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
                train_path, test_path
            )

            # Step 3: Model Training
            model_trainer = ModelTrainer()
            roc_auc = model_trainer.initiate_model_trainer(train_arr, test_arr)

            logging.info("Training pipeline completed successfully")

            return roc_auc

        except Exception as e:
            raise CustomException(e, sys)
        


if __name__ == "__main__":
    obj = TrainPipeline()
    score = obj.run_pipeline()
    print("Final ROC-AUC:", score)