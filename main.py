from fastapi import FastAPI, UploadFile, File, Form
from api.code import data_analysis, data_visualization, data_preprocessing, feature_importance, model_apply, predict
import os
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/static", StaticFiles(directory="frontend"), name = "static")

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
    
# @app.post("/start")
# def start(file:UploadFile File(...)):
#     try:


@app.get("/data_analysis")
def analysis():
    return data_analysis()

@app.get("/data_visualization")
def visualization():
    return data_visualization()

@app.get("/data_preprocessing")
def preprocessing():
    return data_preprocessing()

@app.get("/feature_importance")
def importance():
    return feature_importance()

@app.get("/model_apply")
def apply_model():
    return model_apply()

@app.get("/predict")
def make_prediction():
    return predict()



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)


    
