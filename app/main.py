from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Load the model and vectorizer
model = pickle.load(open("model/spam_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

app = FastAPI()

# Define input structure
class Message(BaseModel):
    message: str

# Define prediction route
@app.post("/predict")
def predict(data: Message):
    vector = vectorizer.transform([data.message])
    prediction = model.predict(vector)
    return {"prediction": "Spam" if prediction[0] == 1 else "Not Spam"}
