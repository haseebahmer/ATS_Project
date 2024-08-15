import subprocess
from heart_disease_prediction.entity.config_entity import DataIngestionConfig
from heart_disease_prediction.components.data_ingestion import DataIngestion
from heart_disease_prediction.components.data_transformation import DataTransformation


class TrainingPipeline:
    def __init__(self, schema_path: str):
        self.schema_path = schema_path

    def run_data_ingestion(self):
        # Data Ingestion
        data_ingestion = DataIngestion(DataIngestionConfig)
        diArtifacts = data_ingestion.initiate_data_ingestion()
        print("Data ingestion completed.")
        return diArtifacts

    def run_data_validation(self, test_csv_path: str):
        # Data Validation
        subprocess.run(
            [
                "python",
                "heart_disease_prediction/components/data_validation.py",
                test_csv_path,
                self.schema_path,
            ]
        )
        print("Data validation completed.")

    def run_data_transformation(self, diArtifacts):
        # Data Transformation
        data_transformation = DataTransformation(diArtifacts, self.schema_path)
        transformation_artifacts = data_transformation.initiate_data_transformation()
        print("Data transformation completed. Artifacts:", transformation_artifacts)
        return transformation_artifacts

    def train(self):
        # Run data ingestion
        diArtifacts = self.run_data_ingestion()

        # Run data validation
        self.run_data_validation(diArtifacts.test_file_path)

        # Run data transformation
        transformation_artifacts = self.run_data_transformation(diArtifacts)

        # Return paths for later use
        return {
            "model_path": "./artifact/best_model.pkl",
            "preprocessing_path": "./artifact/preprocessing.pkl",
            "test_data_path": diArtifacts.test_file_path,
        }