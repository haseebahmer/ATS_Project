
import os
import sys
import pandas as pd
from dotenv import load_dotenv
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from pymongo import MongoClient
import certifi
from heart_disease_prediction.entity.config_entity import DataIngestionConfig
from heart_disease_prediction.entity.artifact_entity import DataIngestionArtifact

from heart_disease_prediction.constants import DATABASE_NAME, COLLECTION_NAME

sys.path.append(os.path.abspath("E:\ATS_Project"))


# Load environment variables
load_dotenv()

ca = certifi.where()

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
            self.dataframe = None
        except Exception as e:
            raise Exception("An error occurred in the constructor") from e

    def read_data_from_db(self) -> None:
        """
        Reads Data from the MongoDB database
        """
        try:
            mongo_url = os.getenv("MONGODB_URL")
            if not mongo_url:
                raise ValueError("MONGODB_URL environment variable not set")

            mongo_client = MongoClient(mongo_url, tlsCAFile=ca)

            db = mongo_client[DATABASE_NAME]
            collection = db[COLLECTION_NAME]

            data = list(collection.find())
            self.dataframe = pd.DataFrame(data)

        except Exception as e:
            raise Exception("An error occurred while reading data from the database") from e

    def export_data_into_feature_store(self) -> DataFrame:
        """
        Method Name :   export_data_into_feature_store
        Description :   This method exports data from MongoDB to a CSV file

        Output      :   Data is returned as a DataFrame
        On Failure  :   Writes an exception log and then raises an exception
        """
        try:
            self.read_data_from_db()

            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            self.dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return self.dataframe

        except Exception as e:
            raise Exception("An error occurred while exporting data to the feature store") from e

    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        """
        Method Name :   split_data_as_train_test
        Description :   This method splits the DataFrame into train set and test set based on split ratio

        Output      :   Train and test sets are saved to the respective file paths
        On Failure  :   Writes an exception log and then raises an exception
        """
        try:
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)

        except Exception as e:
            raise Exception("An error occurred while splitting data into train and test sets") from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Method Name :   initiate_data_ingestion
        Description :   This method initiates the data ingestion components of the training pipeline

        Output      :   Train set and test set are returned as the artifacts of data ingestion components
        On Failure  :   Writes an exception log and then raises an exception
        """
        try:
            dataframe = self.export_data_into_feature_store()
            self.split_data_as_train_test(dataframe)

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
                feature_store_path=self.data_ingestion_config.feature_store_file_path,
            )

            return data_ingestion_artifact

        except Exception as e:
            raise Exception("An error occurred during data ingestion") from e
