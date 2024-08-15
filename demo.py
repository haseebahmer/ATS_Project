


from heart_disease_prediction.entity.config_entity import DataIngestionConfig
#from heart_disease_prediction.entity.artifact_entity import DataIngestionArtifact
from heart_disease_prediction.components.data_ingestion import DataIngestion
from dotenv import load_dotenv

import os

from heart_disease_prediction.components.data_transformation import DataTransformation

# # Load environment variables from the .env file
load_dotenv()

# # Data Ingestion
data_ingestion = DataIngestion(DataIngestionConfig)
diArtifacts = data_ingestion.initiate_data_ingestion()


# # Extract the test.csv path and schema.yaml path



from heart_disease_prediction.components.data_validation import (
    load_csv_data,
    load_schema,
    DataValidator
)

# Extract the test.csv path and schema.yaml path
test_csv_path = diArtifacts.test_file_path  # Ensure diArtifacts is defined
schema_path = "./config/schema.yaml"


def validate_data(file_path, schema_path):
    data = load_csv_data(file_path)
    schema = load_schema(schema_path)

    print("Loaded data from:", file_path)
    print(data.head())  # Print the first few rows of the data

    validator = DataValidator(data, schema)
    validator.validate()
    validator.summary()

if __name__ == "__main__":
    validate_data(test_csv_path, schema_path)



# Data Transformation
data_transformation = DataTransformation(diArtifacts, schema_path)
transformation_artifacts = data_transformation.initiate_data_transformation()

print("Data transformation completed. Artifacts:", transformation_artifacts)



from heart_disease_prediction.components.model_trainer import (
    read_transformed_data,
    model_evaluation,
)

import os
import sys



# Read transformed data and evaluate the model
read_transformed_data()
expected_score = 0.85  # Define the expected accuracy score
model_evaluation(expected_score)

print("Model evaluation completed.")

# Paths to your files
model_path = "./artifact/best_model.pkl"
preprocessing_path = "./artifact/preprocessing.pkl"
test_data_path = diArtifacts.test_file_path




import pickle
import pandas as pd
from sklearn.metrics import accuracy_score

def load_object(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def estimator(model_path, preprocessor_path, user_data):
    preprocessor = load_object(preprocessor_path)
    model = load_object(model_path)

    # Convert user data to DataFrame
    user_df = pd.DataFrame([user_data])

    # Ensure columns are in the correct order and fill missing ones
    required_columns = [
        "age", "hypertension", "heart_disease", "avg_glucose_level", "bmi",
        "gender", "ever_married", "work_type", "Residence_type", "smoking_status"
    ]
    for col in required_columns:
        if col not in user_df.columns:
            user_df[col] = 'unknown' if col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'] else 0

    user_df = user_df[required_columns]

    # Apply preprocessing
    X_transformed = preprocessor.transform(user_df)

    # Make prediction
    y_pred = model.predict(X_transformed)
    return y_pred[0]

if __name__ == "__main__":
    # Paths to your model and preprocessing files
    model_path = "./artifact/best_model.pkl"
    preprocessor_path = "./artifact/preprocessing.pkl"

    # Get user input
    print("Please enter the following details:")
    user_data = {
        "age": input("Enter age: "),
        "hypertension": input("Enter hypertension (0 for No, 1 for Yes): "),
        "heart_disease": input("Enter heart disease (0 for No, 1 for Yes): "),
        "avg_glucose_level": input("Enter average glucose level (e.g., 85.96): "),
        "bmi": input("Enter BMI: "),
        "gender": input("Enter gender (Male/Female): "),
        "ever_married": input("Ever married (Yes/No): "),
        "work_type": input("Work type (Private/Self-employed/Govt_job/Children/Never_worked): "),
        "Residence_type": input("Residence type (Urban/Rural): "),
        "smoking_status": input("Smoking status (formerly smoked/never smoked/smokes/Unknown): ")
    }

    # Convert numeric fields to appropriate types
    numeric_columns = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
    for column in numeric_columns:
        user_data[column] = pd.to_numeric(user_data[column], errors='coerce')

    # Predict whether the user has a stroke
    prediction = estimator(model_path, preprocessor_path, user_data)
    print(f"Prediction: {'Stroke' if prediction == 1 else 'No Stroke'}")


