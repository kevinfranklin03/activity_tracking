import joblib
import os
import numpy as np

def init():
    global model
    # Load the model from the Azure ML Model directory
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "trained_model")
    model = joblib.load(model_path)
    print("Model loaded successfully.")

def run(data):
    try:
        # Parse input data
        input_data = np.array(data.get("data"))
        # Make predictions
        predictions = model.predict(input_data)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        return {"error": str(e)}
