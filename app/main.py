import pickle
from fastapi import FastAPI, Request
from pydantic import BaseModel

# Load model and vectorizer
with open("model/spam_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

app = FastAPI()

class Message(BaseModel):
    text: str

@app.post("/predict")
def predict(message: Message):
    vector = vectorizer.transform([message.text])
    prediction = model.predict(vector)
    return {"prediction": "spam" if prediction[0] == 1 else "not spam"}
