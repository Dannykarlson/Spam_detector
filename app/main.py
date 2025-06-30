# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os

# Define a Pydantic model for request validation
class Message(BaseModel):
    text: str

# Initialize FastAPI app
app = FastAPI()

# Load the saved model and vectorizer
model_path = os.path.join("model", "spam_model.pkl")
vectorizer_path = os.path.join("model", "vectorizer.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

# Define a simple route to check app status
@app.get("/")
def read_root():
    return {"message": "Spam Detector API is live!"}

# Define prediction endpoint
@app.post("/predict")
def predict(data: Message):
    # Transform the text using the loaded vectorizer
    vect_text = vectorizer.transform([data.text])
    
    # Make prediction
    prediction = model.predict(vect_text)
    
    # Return result
    result = "Spam" if prediction[0] == 1 else "Not Spam"
    return {"prediction": result}
