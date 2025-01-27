from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from api.code import data_analysis, data_visualization, data_preprocessing, feature_importance, model_apply, predict
import os
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import joblib 
from sklearn.preprocessing import MinMaxScaler
import pandas as pd 




app = FastAPI()

app.mount("/static", StaticFiles(directory="frontend"), name = "static")
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
app.mount("/files", StaticFiles(directory="H:/Cardiovascular_Disease_Prediction/files"), name="files")
app.mount("/data/processed", StaticFiles(directory="H:/Cardiovascular_Disease_Prediction/data/processed"), name="processed-data")

# Serve home.html
@app.get("/", response_class=HTMLResponse)
def index():
    with open("frontend/home.html", "r", encoding="utf-8") as file:
        return file.read()
    
@app.post("/start")
async def upload_merge(file: UploadFile = File(...)):
    try:
        # Define user-specific directories
        files_dir = os.path.join("data", "raw")

        # Ensure the directory exists
        os.makedirs(files_dir, exist_ok=True)

        # Save the uploaded file as merge_file.csv
        file_path = os.path.join(files_dir, "cardio_data.csv")
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Return the success message with file path
        return {
            "message": "File uploaded successfully!",
            "cardio_data": file_path
        }

    except Exception as e:
        # Handle unexpected errors
        return {
            "message": "Error in uploading file.",
            "error": str(e)
        }

@app.post("/data_analysis")
def analysis():
    return data_analysis()

    
@app.get("/data_visualization", response_class=HTMLResponse)
def visualization():
    return data_visualization()

@app.get("/feature_importance")
def importance():
    return feature_importance() 


@app.get("/data_preprocessing", response_class=HTMLResponse)
def preprocessing():
    # Call the preprocessing function
    data_preprocessing()
    file_Path = "files/final_data.csv"

    return file_Path


@app.get("/model_apply")
def apply_model():
    return model_apply()




@app.post("/predict")
def make_prediction(input_data: dict):
    """
    json_input = {
    "ap_hi": 120,
    "ap_lo": 80,
    "cholesterol": 1, 
    "age_years": 47,  
    "bmi": 26.573129         
}

    API endpoint to call the predict function from code.py
    :param input_data: JSON with input features
    :return: Prediction result
    """
    return predict(input_data)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)


    
