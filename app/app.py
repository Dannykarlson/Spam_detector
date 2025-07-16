import streamlit as st
import pickle

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.set_page_config(page_title="Spam Detector", layout="centered")
st.title("📧 Spam Detector")

input_text = st.text_area("Enter a message:", height=150)

if st.button("Check if Spam"):
    if input_text.strip() == "":
        st.warning("Please enter a message first.")
    else:
        text_vector = vectorizer.transform([input_text])
        prediction = model.predict(text_vector)[0]
        
        if prediction == 1:
            st.error("🚨 This message is SPAM!")
        else:
            st.success("✅ This message is NOT spam.")
