from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from api.code import preprocess_data, train_model, predict
import shutil
import os

app = FastAPI()

# Welcome Page
@app.get("/")
def index():
    with open("frontend/index.html", "r") as file:
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

# Preprocess API
@app.post("/preprocess")
async def preprocess(file: UploadFile = File(...)):
    raw_data_path = f"data/raw/{file.filename}"
    processed_data_path = "data/processed/Final_Dataset.csv"

    # Save uploaded file
    os.makedirs("data/raw", exist_ok=True)
    with open(raw_data_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Preprocess the data
    result = preprocess_data(raw_data_path, processed_data_path)
    return result

# Train Model API
@app.get("/train")
def train():
    processed_file = "data/processed/Final_Dataset.csv"
    model_path = "model/model.pkl"
    result = train_model(processed_file, model_path)
    return result

# Predict API
@app.post("/predict")
def predict_api(input_data: dict):
    model_path = "model/model.pkl"
    result = predict(model_path, input_data)
    return result



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)


    
