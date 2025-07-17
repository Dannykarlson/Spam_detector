# streamlit_app.py
import streamlit as st
import requests

st.title("📩 Spam Detector")
message = st.text_area("Enter a message to check:")

if st.button("Check if Spam"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        response = requests.post(
            "https://your-backend-url.onrender.com/predict",  # Update this later
            json={"text": message}
        )
        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.success(f"This is **{prediction.upper()}**")
        else:
            st.error("Failed to get prediction from API.")
