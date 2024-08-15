import os
import pandas as pd
import yaml
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from heart_disease_prediction.entity.config_entity import DataIngestionConfig
from heart_disease_prediction.entity.artifact_entity import DataIngestionArtifact
from heart_disease_prediction.constants import ARTIFACT_DIR
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    transformed_train_dir: str = os.path.join(ARTIFACT_DIR, "transformed_train")
    transformed_test_dir: str = os.path.join(ARTIFACT_DIR, "transformed_test")
    preprocessing_object_path: str = os.path.join(ARTIFACT_DIR, "preprocessing.pkl")

class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, schema_path: str):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.schema_path = schema_path
        self.config = DataTransformationConfig()

    def load_data(self, file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)

    def read_schema(self):
        with open(self.schema_path, "r") as file:
            schema = yaml.safe_load(file)
        return schema

    def get_preprocessor(self, schema: dict):
        num_features = schema.get("num_features", [])
        oh_columns = schema.get("oh_columns", [])
        transform_columns = schema.get("transform_columns", [])

        num_pipeline = Pipeline([
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler())
        ])

        oh_pipeline = Pipeline([
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        power_pipeline = Pipeline([
            ("imputer", KNNImputer(n_neighbors=5)),
            ("power", PowerTransformer())
        ])

        preprocessor = ColumnTransformer([
            ("num", num_pipeline, num_features),
            ("cat", oh_pipeline, oh_columns),
            ("power", power_pipeline, transform_columns),
        ])

        return preprocessor

    def initiate_data_transformation(self):
        schema = self.read_schema()

        train_df = self.load_data(self.data_ingestion_artifact.trained_file_path)
        test_df = self.load_data(self.data_ingestion_artifact.test_file_path)

        preprocessor = self.get_preprocessor(schema)

        drop_columns = schema.get("drop_columns", [])
        X_train = train_df.drop(columns=drop_columns)
        y_train = train_df["stroke"]
        X_test = test_df.drop(columns=drop_columns)
        y_test = test_df["stroke"]

        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)

        os.makedirs(self.config.transformed_train_dir, exist_ok=True)
        os.makedirs(self.config.transformed_test_dir, exist_ok=True)

        # Save transformed data as CSV files
        transformed_train_path = os.path.join(self.config.transformed_train_dir, "train_transformed.csv")
        transformed_test_path = os.path.join(self.config.transformed_test_dir, "test_transformed.csv")

        # Ensure column names are consistent
        train_df_transformed = pd.DataFrame(X_train_resampled, columns=[f"feature_{i}" for i in range(X_train_resampled.shape[1])])
        train_df_transformed["stroke"] = y_train_resampled

        test_df_transformed = pd.DataFrame(X_test_transformed, columns=[f"feature_{i}" for i in range(X_test_transformed.shape[1])])
        test_df_transformed["stroke"] = y_test

        train_df_transformed.to_csv(transformed_train_path, index=False)
        test_df_transformed.to_csv(transformed_test_path, index=False)

        # Save the preprocessing object
        with open(self.config.preprocessing_object_path, 'wb') as file:
            pickle.dump(preprocessor, file)

        return {
            "transformed_train_path": transformed_train_path,
            "transformed_test_path": transformed_test_path,
            "preprocessing_object_path": self.config.preprocessing_object_path,
        }

if __name__ == "__main__":
    data_ingestion_artifact = DataIngestionArtifact(
        trained_file_path="train.csv",
        test_file_path="test.csv",
        feature_store_path="feature_store",
    )

    schema_path = "./config/schema.yaml"
    data_transformation = DataTransformation(data_ingestion_artifact, schema_path)
    artifacts = data_transformation.initiate_data_transformation()

    print("Data transformation completed. Artifacts:", artifacts)
