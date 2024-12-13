import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from azureml.core.run import Run
from azureml.core import Workspace, Model

# Get Azure ML Run Context
run = Run.get_context()

def main(data_path, model_path, sample_frac=0.1):
    # Columns from dataset to load
    columns_to_load = [
        "T_xacc", "T_yacc", "T_zacc", "T_xgyro", "T_ygyro", "T_zgyro", "T_xmag", "T_ymag", "T_zmag",
        "RA_xacc", "RA_yacc", "RA_zacc", "RA_xgyro", "RA_ygyro", "RA_zgyro", "RA_xmag", "RA_ymag", "RA_zmag",
        "LA_xacc", "LA_yacc", "LA_zacc", "LA_xgyro", "LA_ygyro", "LA_zgyro", "LA_xmag", "LA_ymag", "LA_zmag",
        "RL_xacc", "RL_yacc", "RL_zacc", "RL_xgyro", "RL_ygyro", "RL_zgyro", "RL_xmag", "RL_ymag", "RL_zmag",
        "LL_xacc", "LL_yacc", "LL_zacc", "LL_xgyro", "LL_ygyro", "LL_zgyro", "LL_xmag", "LL_ymag", "LL_zmag"
    ]

    # Load dataset in chunks
    chunks = []
    for chunk in pd.read_csv(data_path, usecols=columns_to_load, chunksize=10000, dtype=np.float32):
        chunks.append(chunk)
    data = pd.concat(chunks, ignore_index=True)

    # Take a sample of the data
    data_sample = data.sample(frac=sample_frac, random_state=42)

    # Split features and labels
    X = data_sample.iloc[:, :]  # Features
    y = np.random.randint(0, 2, size=len(data_sample))  # Randomly generated binary labels for example

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Logistic Regression model
    model = LogisticRegression(random_state=42, max_iter=500)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Model Accuracy: {accuracy}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Log metrics to Azure ML
    run.log("model_type", "LogisticRegression")
    run.log("accuracy", accuracy)

    # Save the model locally
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")

    # Register the model in Azure ML
    register_model(model_path)

    # Complete the run
    run.complete()

def register_model(model_path, model_name="activity_recognition_model"):
    # Get the Azure ML workspace from the run context
    ws = run.experiment.workspace

    # Register the model
    model = Model.register(
        workspace=ws,
        model_path=model_path,
        model_name=model_name,
        tags={"type": "logistic_regression"},
        description="Logistic regression model for activity recognition."
    )
    print(f"Model registered: {model.name}, Version: {model.version}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to save the model")
    parser.add_argument("--sample_frac", type=float, default=0.1, help="Fraction of data to sample")
    args = parser.parse_args()

    main(args.data_path, args.model_path, args.sample_frac)
