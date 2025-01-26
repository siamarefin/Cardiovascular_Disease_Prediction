import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.preprocessing import MinMaxScaler

def data_analysis():
    # Example analysis logic
    return {"message": "Data analysis complete!"}

def data_visualization():
    # Example visualization logic``
    return {"message": "Data visualizations generated!"}

def data_preprocessing():
    # Example preprocessing logic
    return {"message": "Data preprocessing complete!"}

def feature_importance():
    # Example feature importance logic
    return {"message": "Feature importance identified!"}

def model_apply():
    # Example model training logic
    return {"message": "Model trained successfully!"}

# Load the model and scaler
model_path = "model/model.pkl"
try:
    best_model = joblib.load(model_path)  # Load the trained model
except Exception as e:
    raise Exception(f"Error loading model or scaler: {str(e)}")

def predict(input_data: dict):
    """
    Predict the outcome based on input data using the pre-trained model.
    :param input_data: Dictionary containing input features.
    :return: Dictionary with prediction and probabilities.
    """
    # Convert JSON input to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure all required columns are present
    required_columns = ["ap_hi", "ap_lo", "cholesterol", "age_years", "bmi"]
    for col in required_columns:
        if col not in input_df.columns:
            return {"error": f"Missing required field: {col}"}

    try:
        # Normalize the input data using the same scaler as in training

        # Make predictions
        prediction = best_model.predict(input_df)

        # Return the result as a dictionary
        return {
            "input": input_data,
            "predicted_cardio": int(prediction[0]),  # 0: No, 1: Yes
        }
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}
