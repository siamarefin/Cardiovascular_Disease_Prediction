import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def preprocess_data(file_path: str, output_path: str):
    """Preprocess the raw data and save it to the processed folder."""
    df = pd.read_csv(file_path)
    # Example preprocessing logic
    df['bmi'] = df['weight'] / (df['height'] / 100) ** 2  # Calculate BMI
    df.to_csv(output_path, index=False)
    return {"message": "Data preprocessing complete!", "processed_file": output_path}

def train_model(processed_file: str, model_path: str):
    """Train a model and save it to the model folder."""
    df = pd.read_csv(processed_file)
    X = df.drop(columns=["cardio"])
    y = df["cardio"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, model_path)
    return {"message": "Model training complete!", "model_path": model_path}

def predict(model_path: str, input_data: dict):
    """Load the model and make a prediction."""
    model = joblib.load(model_path)
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    return {"prediction": int(prediction)}
