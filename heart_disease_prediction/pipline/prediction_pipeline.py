import pickle
import pandas as pd
from heart_disease_prediction.components import estimator

def load_object(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def run_prediction_pipeline(model_path, preprocessing_path, user_data):
    # Ensure all columns are present
    required_columns = [
        "age", "hypertension", "heart_disease", "avg_glucose_level", "bmi",
        "gender", "ever_married", "work_type", "Residence_type", "smoking_status"
    ]
    user_df = pd.DataFrame([user_data])
    
    # Fill missing columns with default values
    for col in required_columns:
        if col not in user_df.columns:
            user_df[col] = 'unknown' if col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'] else 0

    user_df = user_df[required_columns]

    # Load preprocessor and model
    preprocessor = load_object(preprocessing_path)
    model = load_object(model_path)

    # Apply preprocessing
    X_transformed = preprocessor.transform(user_df)

    # Make prediction
    prediction = model.predict(X_transformed)
    
    result =  'Stroke' if prediction[0] == 1 else 'No Stroke'

    return result


if __name__ == "__main__": 
    run_prediction_pipeline()