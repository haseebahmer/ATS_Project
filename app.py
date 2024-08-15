from heart_disease_prediction.pipline.training_pipeline import run_training_pipeline




    # Run the training pipeline
#run_training_pipeline()


from heart_disease_prediction.pipline.prediction_pipeline import run_prediction_pipeline

def main():
    # Paths to your model and preprocessing files
    model_path = "./artifact/best_model.pkl"
    preprocessing_path = "./artifact/preprocessing.pkl"
    
    # Run the prediction pipeline
    run_prediction_pipeline(model_path, preprocessing_path)

if __name__ == "__main__":
    main()

