import subprocess
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os

from heart_disease_prediction.pipline.prediction_pipeline import run_prediction_pipeline

app = FastAPI()

class PredictionInput(BaseModel):
    age: float
    hypertension: int
    heart_disease: int
    avg_glucose_level: float
    bmi: float
    gender: str
    ever_married: str
    work_type: str
    Residence_type: str
    smoking_status: str

@app.get("/", response_class=HTMLResponse)
async def home():
    html_content = """
    <html>
    <head>
        <title>Heart Disease Prediction</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            form { margin-bottom: 20px; }
            label { display: block; margin: 5px 0; }
            input[type="text"] { width: 100%; padding: 8px; margin-bottom: 10px; }
            button { padding: 10px 20px; background-color: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background-color: #45a049; }
            .container { max-width: 800px; margin: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Heart Disease Prediction</h1>
            <form action="/train" method="post">
                <button type="submit">Train Model</button>
            </form>
            <form action="/predict" method="post">
                <label for="age">Age:</label><input type="text" id="age" name="age">
                <label for="hypertension">Hypertension:</label><input type="text" id="hypertension" name="hypertension">
                <label for="heart_disease">Heart Disease:</label><input type="text" id="heart_disease" name="heart_disease">
                <label for="avg_glucose_level">Average Glucose Level:</label><input type="text" id="avg_glucose_level" name="avg_glucose_level">
                <label for="bmi">BMI:</label><input type="text" id="bmi" name="bmi">
                <label for="gender">Gender:</label><input type="text" id="gender" name="gender">
                <label for="ever_married">Ever Married:</label><input type="text" id="ever_married" name="ever_married">
                <label for="work_type">Work Type:</label><input type="text" id="work_type" name="work_type">
                <label for="Residence_type">Residence Type:</label><input type="text" id="Residence_type" name="Residence_type">
                <label for="smoking_status">Smoking Status:</label><input type="text" id="smoking_status" name="smoking_status">
                <button type="submit">Predict</button>
            </form>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/train")
async def train():
    try:
        # Run the training pipeline script and capture the output
        result = subprocess.run(["python", "heart_disease_prediction/pipline/training_pipeline.py"], capture_output=True, text=True)
        
        # File paths
        train_csv_path = "./artifact/08_06_2024_22_32_52/data_ingestion/ingested/train.csv"
        test_transformed_csv_path = "./artifact/transformed_test/test_transformed.csv"
        transformed_train_csv_path = "./artifact/transformed_train/train_transformed.csv"
        
        # Read the contents of the files
        train_df = pd.read_csv(train_csv_path).head(10)
        test_transformed_df = pd.read_csv(test_transformed_csv_path).head(10)
        transformed_train_df = pd.read_csv(transformed_train_csv_path).head(10)
        
        # Convert DataFrames to HTML
        train_html = train_df.to_html(classes="data", header=True, index=False)
        test_transformed_html = test_transformed_df.to_html(classes="data", header=True, index=False)
        transformed_train_html = transformed_train_df.to_html(classes="data", header=True, index=False)

        # Return the HTML response
        if result.returncode == 0:
            return HTMLResponse(f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                    .container {{ max-width: 800px; margin: auto; }}
                    .data {{ border-collapse: collapse; width: 100%; }}
                    .data th, .data td {{ border: 1px solid #ddd; padding: 8px; }}
                    .data th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Training completed successfully!</h1>
                    <h2>Train Data:</h2>
                    {train_html}
                    <h2>Test Transformed Data:</h2>
                    {test_transformed_html}
                    <h2>Transformed Train Data:</h2>
                    {transformed_train_html}
                    <pre>{result.stdout}</pre>
                </div>
            </body>
            </html>
            """)
        else:
            return HTMLResponse(f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #d9534f; }}
                    pre {{ background-color: #f8d7da; color: #721c24; padding: 10px; border: 1px solid #f5c6cb; }}
                </style>
            </head>
            <body>
                <h1>Error occurred during training</h1>
                <pre>{result.stderr}</pre>
            </body>
            </html>
            """)
    except Exception as e:
        return HTMLResponse(f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #d9534f; }}
                pre {{ background-color: #f8d7da; color: #721c24; padding: 10px; border: 1px solid #f5c6cb; }}
            </style>
        </head>
        <body>
            <h1>Error occurred: {e}</h1>
        </body>
        </html>
        """)

@app.post("/predict")
async def predict(request: Request):
    form = await request.form()
    input_data = {
        "age": float(form.get("age")),
        "hypertension": int(form.get("hypertension")),
        "heart_disease": int(form.get("heart_disease")),
        "avg_glucose_level": float(form.get("avg_glucose_level")),
        "bmi": float(form.get("bmi")),
        "gender": form.get("gender"),
        "ever_married": form.get("ever_married"),
        "work_type": form.get("work_type"),
        "Residence_type": form.get("Residence_type"),
        "smoking_status": form.get("smoking_status")
    }

    try:
        # Debug: print input data
        print("Input Data:", input_data)

        model_path = "./artifact/best_model.pkl"
        preprocessing_path = "./artifact/preprocessing.pkl"

        # Run prediction
        prediction_result = run_prediction_pipeline(model_path, preprocessing_path, input_data)
        
        # Debug: print prediction result
        print("Prediction Result:", prediction_result)

        return HTMLResponse(f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
            </style>
        </head>
        <body>
            <h1>Prediction Result:</h1>
            <p>{prediction_result}</p>
        </body>
        </html>
        """)
    except Exception as e:
        # Debug: print error details
        print("Error:", e)
        return HTMLResponse(f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #d9534f; }}
                pre {{ background-color: #f8d7da; color: #721c24; padding: 10px; border: 1px solid #f5c6cb; }}
            </style>
        </head>
        <body>
            <h1>Error occurred: {e}</h1>
        </body>
        </html>
        """)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
