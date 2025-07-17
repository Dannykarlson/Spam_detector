import os
import pickle

# Get current file path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model
with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)

# Load vectorizer
with open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)
