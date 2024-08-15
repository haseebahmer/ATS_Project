
import os
from dotenv import load_dotenv
import certifi
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from pymongo import MongoClient
from heart_disease_prediction.entity.config_entity import DataIngestionConfig
from heart_disease_prediction.entity.artifact_entity import DataIngestionArtifact
from heart_disease_prediction.constants import DATABASE_NAME, COLLECTION_NAME

ca = certifi.where()


class DataIngestion:
    def __init__(
        self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()
    ):
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
            self.dataframe = None  # Initialize the dataframe attribute
        except Exception as e:
            raise Exception(f"An error occurred in the constructor: {e}")

    def read_data_from_db(self) -> None:
        """
        Reads data from the MongoDB database and sets it as an instance variable.
        """
        try:
            # Fetch the MongoDB URL from environment variables
            mongo_url = os.getenv("MONGODB_URL")
            if not mongo_url:
                raise ValueError("MONGODB_URL environment variable not set.")

            # MongoDB connection details
            mongo_client = MongoClient(mongo_url, tlsCAFile=ca)

            db = mongo_client[DATABASE_NAME]  # it is heart disease
            collection = db[COLLECTION_NAME]  # it is data stroke

            # Fetch all data from the collection
            data = list(collection.find())

            # Convert to DataFrame
            self.dataframe = pd.DataFrame(data)

            # Optionally, remove MongoDB's default `_id` field if it exists
            if "_id" in self.dataframe.columns:
                self.dataframe = self.dataframe.drop(columns=["_id"])
        except Exception as e:
            raise Exception(f"An error occurred while reading data from db: {e}")

    def export_data_into_feature_store(self) -> DataFrame:
        """
        Exports data from MongoDB to a CSV file.
        """
        try:
            # logging.info(f"Exporting data from mongodb")
            self.read_data_from_db()
            # logging.info(f"Shape of dataframe: {self.dataframe.shape}")

            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            # logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")

            self.dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return self.dataframe
        except Exception as e:
            raise Exception(
                f"An error occurred while exporting data to feature store: {e}"
            )

    def split_data_as_train_test(self) -> None:
        """
        Splits the dataframe into train set and test set based on the split ratio.
        """
        # logging.info("Entered split_data_as_train_test method of Data_Ingestion class")

        try:
            train_set, test_set = train_test_split(
                self.dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
            )
            # logging.info("Performed train test split on the dataframe")
            # logging.info("Exited split_data_as_train_test method of Data_Ingestion class")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            # logging.info(f"Exporting train and test file path.")
            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )
            # logging.info(f"Exported train and test file path.")
        except Exception as e:
            raise Exception(f"An error occurred while splitting data: {e}")

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Initiates the data ingestion components of the training pipeline.
        """
        # logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")

        try:
            self.export_data_into_feature_store()
            # logging.info("Got the data from mongodb")

            self.split_data_as_train_test()
            # logging.info("Performed train test split on the dataset")
            # logging.info("Exited initiate_data_ingestion method of Data_Ingestion class")

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
                feature_store_path=self.data_ingestion_config.feature_store_file_path,
            )
            # logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise Exception(f"An error occurred while initiating data ingestion: {e}")