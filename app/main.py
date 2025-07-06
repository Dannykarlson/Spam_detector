
from fastapi import FastAPI
from pydantic import BaseModel
import os
import requests
import pickle

app = FastAPI()

# Function to download files if they don't exist
def download_file(url, filepath):
    if not os.path.exists(filepath):
        print(f"Downloading {filepath}...")
        response = requests.get(url)
        with open(filepath, "wb") as f:
            f.write(response.content)

# Direct download links from Google Drive
MODEL_URL = "https://drive.google.com/uc?export=download&id=149SgoI94kO75BB2bM8hzKkf8ZeTNW_cF"
VECTORIZER_URL = "https://drive.google.com/uc?export=download&id=1q_WjrCHOXExRBzFGGI2g69-LMzqs9oRO"

# File paths to store the downloaded files
model_path = "model/spam_model.pkl"
vectorizer_path = "model/vectorizer.pkl"

# Download the files
download_file(MODEL_URL, model_path)
download_file(VECTORIZER_URL, vectorizer_path)

# Load model and vectorizer
with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

# Define input structure
class Message(BaseModel):
    message: str

# Prediction route
@app.post("/predict")
def predict(data: Message):
    vector = vectorizer.transform([data.message])
    prediction = model.predict(vector)
    return {"prediction": "Spam" if prediction[0] == 1 else "Not Spam"}
