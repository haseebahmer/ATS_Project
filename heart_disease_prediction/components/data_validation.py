import os
import pandas as pd
import sys
import yaml
def load_file(file_path, file_type="csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"file does not exist.")
    if file_type == "csv":
        return pd.read_csv(file_path)
    elif file_type == "yaml":
        with open(file_path, "r") as file:
            return yaml.safe_load(file)

class DataValidator:
    def __init__(self, data, schema):
        self.data = data
        self.schema = schema

    def validate_columns(self):
        expected_columns = self.schema["columns"]
        actual_columns = self.data.columns.tolist()
        
        # Check if column names match
        columns_match = expected_columns == actual_columns
        
        # Check if the number of columns match
        num_columns_match = len(expected_columns) == len(actual_columns)
        
        return columns_match, num_columns_match, expected_columns, actual_columns

    def validate_numerical_columns(self):
        if "numerical_columns" not in self.schema:
            raise KeyError("numerical_columns key not found in schema")
        expected_columns = self.schema["numerical_columns"]
        actual_columns = self.data.select_dtypes(include=["number"]).columns.tolist()
        return expected_columns == actual_columns, expected_columns, actual_columns

    def validate_categorical_columns(self):
        if "categorical_columns" not in self.schema:
            raise KeyError("categorical_columns key not found in schema")
        expected_columns = self.schema["categorical_columns"]
        actual_columns = self.data.select_dtypes(include=["object"]).columns.tolist()
        return expected_columns == actual_columns, expected_columns, actual_columns

    def validate(self):
        checks = {
            "Column Names": self.validate_columns(),
            "Numerical Columns": self.validate_numerical_columns(),
            "Categorical Columns": self.validate_categorical_columns()
        }

        for check, result in checks.items():
            status, expected, found = result
            if status:
                print(f"{check} check passed.")
            else:
                print(f"{check} check failed. Expected: {expected}, Found: {found}")

def validate_data(file_path, schema_path):
    data = load_file(file_path, "csv")
    schema = load_file(schema_path, "yaml")

    print("Loaded data from:", file_path)
    print(data.head())  # Print the first few rows of the data

    validator = DataValidator(data, schema)
    validator.validate()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_validation.py <path_to_csv> <path_to_schema>")
    else:
        validate_data(sys.argv[1], sys.argv[2])
