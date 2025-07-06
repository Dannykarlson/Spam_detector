from fastapi import FastAPI
from pydantic import BaseModel
import os
import requests
import pickle

app = FastAPI()

# Define file download function
def download_file(url, filepath):
    if not os.path.exists(filepath):
        print(f"Downloading {filepath}...")
        response = requests.get(url)
        with open(filepath, "wb") as f:
            f.write(response.content)

# Replace these with your actual direct download links
MODEL_URL = "https://your-link/spam_model.pkl"
VECTORIZER_URL = "https://your-link/vectorizer.pkl"

# Paths to save files
model_path = "model/spam_model.pkl"
vectorizer_path = "model/vectorizer.pkl"

# Download model files if they don't exist
download_file(MODEL_URL, model_path)
download_file(VECTORIZER_URL, vectorizer_path)

# Load the model and vectorizer
model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

# Define input structure
class Message(BaseModel):
    message: str

# Define prediction route
@app.post("/predict")
def predict(data: Message):
    vector = vectorizer.transform([data.message])
    prediction = model.predict(vector)
    return {"prediction": "Spam" if prediction[0] == 1 else "Not Spam"}
