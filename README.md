# Cardiovascular Disease Prediction

This project predicts cardiovascular disease using machine learning models and provides a web interface for data analysis, visualization, and predictions.

---

## Requirements

- Python 3.8+
- FastAPI
- Uvicorn
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- XGBoost
- LightGBM

Install the required libraries using:

```
pip install -r requirements.txt
```

# Usage

## Clone the Repository

```
git clone https://github.com/your-repo/Cardiovascular_Disease_Prediction.git
cd Cardiovascular_Disease_Prediction
```

## Run the Application

```
uvicorn main:app --reload
```

## Docker run

build

```
docker build -t cvd .
```

run

```
docker run --name container -p 8000:8000 cvd
```

## Access the Web Interface : http://127.0.0.1:8000

# Citation

If you use this project, please cite:

Author: Siam Arefin

Email: siam12@student.sust.edu

Github: https://github.com/siamarefin
