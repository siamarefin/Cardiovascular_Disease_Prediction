import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.preprocessing import MinMaxScaler
import os 

def data_analysis():
    # Path to the dataset
    file_path = "data/raw/cardio_data.csv"
    save_path = r"H:\Cardiovascular_Disease_Prediction\files"

    try:
        # Read the CSV
        data = pd.read_csv(file_path)

        # Create a string to store all results
        analysis_results = ""

        # Dataset Overview
        analysis_results += "<h2>Dataset Overview</h2>"
        analysis_results += f"<p>Shape of the dataset: {data.shape}</p>"

        # Columns in the dataset
        analysis_results += "<h3>Columns in the dataset:</h3>"
        analysis_results += f"<p>{', '.join(data.columns)}</p>"

        # Sample data
        analysis_results += "<h3>Sample Data:</h3>"
        analysis_results += data.head().to_html(classes="dataframe", index=False)

        # Statistical summary
        analysis_results += "<h3>Statistical Summary:</h3>"
        analysis_results += data.describe().to_html(classes="dataframe")

        analysis_results += "<h3>Missing Values:</h3>"
        missing_values = data.isnull().sum()
        missing_values_list = missing_values.to_dict()  # Convert to a dictionary for easier iteration

        # Format the missing values as an HTML unordered list
        analysis_results += "<ul>"
        for column, missing in missing_values_list.items():
            analysis_results += f"<li>{column}: {missing}</li>"
        analysis_results += "</ul>"


        # Save the results to an HTML file
        os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
        output_file = os.path.join(save_path, "data_analysis_results.html")

        # Open the file in write mode and save the results
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(analysis_results)

        # Return the path of the saved file along with analysis
        return analysis_results

    except Exception as e:
        # Handle errors
        return f"<h2>Error:</h2><p>{str(e)}</p>"


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
