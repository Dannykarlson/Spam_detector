from fastapi import FastAPI
from pydantic import BaseModel
import gdown
import os
import pickle

# ✅ Download model files if they don't already exist
if not os.path.exists("model/spam_model.pkl"):
    gdown.download(id="149SgoI94kO75BB2bM8hzKkf8ZeTNW_cF", output="model/spam_model.pkl", quiet=False)

if not os.path.exists("model/vectorizer.pkl"):
    gdown.download(id="1q_WjrCHOXExRBzFGGI2g69-LMzqs9oRO", output="model/vectorizer.pkl", quiet=False)

# ✅ Load the model and vectorizer
model = pickle.load(open("model/spam_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# ✅ Create FastAPI app
app = FastAPI()

# ✅ Input format
class Message(BaseModel):
    message: str

# ✅ Prediction endpoint
@app.post("/predict")
def predict(data: Message):
    vector = vectorizer.transform([data.message])
    prediction = model.predict(vector)
    return {"prediction": "Spam" if prediction[0] == 1 else "Not Spam"}
