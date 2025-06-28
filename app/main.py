from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os

# Load the model and vectorizer
model_path = os.path.join("model", "spam_model.pkl")
vectorizer_path = os.path.join("model", "vectorizer.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

# Define the input schema
class Message(BaseModel):
    text: str

# Create FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Spam Detector API"}

@app.post("/predict")
def predict(message: Message):
    data = [message.text]
    transformed = vectorizer.transform(data)
    prediction = model.predict(transformed)
    return {"prediction": "Spam" if prediction[0] == 1 else "Not Spam"}
