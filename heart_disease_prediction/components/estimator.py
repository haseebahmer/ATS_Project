import pickle
import pandas as pd
from sklearn.metrics import accuracy_score

def load_object(file_path):
    """Load a pickled object from a file."""
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def estimator(model_path, preprocessor_path, test_data_path):
    """Load the model and preprocessor, apply transformation on test data, and evaluate accuracy."""
    # Load preprocessor and model
    preprocessor = load_object(preprocessor_path)
    model = load_object(model_path)

    # Load test data
    test_df = pd.read_csv(test_data_path)
    
    # Separate features and target
    X_test = test_df.drop(columns=['stroke'])
    y_test = test_df['stroke']

    # Apply preprocessing transformation
    X_test_transformed = preprocessor.transform(X_test)

    # Make predictions
    y_pred = model.predict(X_test_transformed)
    print("Predictions:", y_pred)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

# Define paths to your files
model_path = "./artifact/best_model.pkl"
preprocessor_path = "./artifact/preprocessing.pkl"
test_data_path = "./artifact/08_08_2024_12_44_30/data_ingestion/ingested/test.csv"  # Update with your actual test data path



# Run the estimator
estimator(model_path, preprocessor_path, test_data_path)




