from fastapi import FastAPI
import pickle
from pydantic import BaseModel

app = FastAPI()

# Load the trained spam model and vectorizer
with open("model/spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Define a request body format
class Message(BaseModel):
    text: str

@app.post("/predict")
def predict_spam(data: Message):
    vector = vectorizer.transform([data.text])
    prediction = model.predict(vector)
    return {"prediction": "Spam" if prediction[0] == 1 else "Not Spam"}
