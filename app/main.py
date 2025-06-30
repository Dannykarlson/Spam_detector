from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Load the model and vectorizer
with open("model/spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Create FastAPI instance
app = FastAPI()

# Create a Pydantic model for the input
class Message(BaseModel):
    message: str

# Define the prediction route
@app.post("/predict")
def predict(data: Message):
    message = [data.message]  # Put message in a list to match vectorizer format
    vect_msg = vectorizer.transform(message)  # Vectorize message
    prediction = model.predict(vect_msg)[0]   # Get prediction (Spam or Not Spam)
    confidence = max(model.predict_proba(vect_msg)[0])  # Highest probability

    return {
        "prediction": prediction,
        "confidence": f"{confidence * 100:.2f}%"
    }
