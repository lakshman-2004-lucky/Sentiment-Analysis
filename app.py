import streamlit as st
from google_drive_downloader import GoogleDriveDownloader as gdd
import tensorflow as tf
import pickle
import os

# Download the model from Google Drive
file_id = "1O2NSkwqNVOAKYJFT3lc0cduP8Enp5uYn"  # Replace with your actual Google Drive file ID
model_path = "./bi_lstm_model.keras"

if not os.path.exists(model_path):  # Download only if not present
    st.write("Downloading model... Please wait.")
    gdd.download_file_from_google_drive(file_id=file_id, dest_path=model_path, unzip=False)

# Load the trained model
st.write("Loading the model...")
model = tf.keras.models.load_model(model_path)
st.write("Model loaded successfully!")

# Load the tokenizer
tokenizer_path = "./tokenizer.pkl"
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

# Function to preprocess text
def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([text])
    return tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=100)  # Adjust maxlen as per training

# Streamlit UI
st.title("Sentiment Analysis with BiLSTM")

user_input = st.text_area("Enter your text:")
if st.button("Predict"):
    if user_input.strip():
        processed_input = preprocess_text(user_input)
        prediction = model.predict(processed_input)
        sentiment = ["Negative", "Neutral", "Positive"]
        st.write(f"Prediction: {sentiment[prediction.argmax()]}")
    else:
        st.write("Please enter some text.")
