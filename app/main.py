from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Define your app
app = FastAPI()

# Load the trained model and vectorizer
with open("model/spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Define input format using Pydantic
class MessageInput(BaseModel):
    message: str

# Define prediction route
@app.post("/predict")
def predict(data: MessageInput):
    message = data.message
    transformed = vectorizer.transform([message])
    prediction = model.predict(transformed)[0]
    return {"prediction": prediction}
