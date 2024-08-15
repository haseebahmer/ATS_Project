from heart_disease_prediction.components.data_ingestion import DataIngestion
from heart_disease_prediction.components.data_transformation import DataTransformation
from heart_disease_prediction.components.data_validation import DataValidator, load_csv_data, load_schema
from heart_disease_prediction.components.model_trainer import read_transformed_data, model_evaluation
from heart_disease_prediction.entity.config_entity import DataIngestionConfig

def run_training_pipeline():
    # Define the path to the schema file
    schema_path = "./config/schema.yaml"  # Adjust the path as needed

    # Data Ingestion
    data_ingestion = DataIngestion(DataIngestionConfig)
    diArtifacts = data_ingestion.initiate_data_ingestion()

    # Data Validation
    test_csv_path = diArtifacts.test_file_path  # Ensure diArtifacts is defined
    data = load_csv_data(test_csv_path)
    schema = load_schema(schema_path)

    print("Loaded data from:", test_csv_path)
    print(data.head())  # Print the first few rows of the data
    validator = DataValidator(data, schema)
    validator.validate()
    validator.summary()

    # Data Transformation
    data_transformation = DataTransformation(diArtifacts, schema_path)
    transformation_artifacts = data_transformation.initiate_data_transformation()
    print("Data transformation completed. Artifacts:", transformation_artifacts)

    # Model Training
    read_transformed_data()
    expected_score = 0.85  # Define the expected accuracy score
    model_evaluation(expected_score)
    print("Model evaluation completed.")

