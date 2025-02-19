import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load saved model
@st.cache_resource
def load_sentiment_model():
    return load_model("bi_lstm_model.keras")

# Load tokenizer
@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as handle:
        return pickle.load(handle)

# Load model and tokenizer
model = load_sentiment_model()
tokenizer = load_tokenizer()

# Define max sequence length (same as during training)
MAX_LEN = 100  # Change this if different during training

# Function to preprocess and predict sentiment
def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding="post")
    prediction = model.predict(padded_sequence)
    sentiment_class = np.argmax(prediction)

    sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_mapping[sentiment_class]

# Streamlit UI
st.title("Sentiment Analysis with Bi-LSTM")
st.write("Enter a text below to predict its sentiment:")

user_input = st.text_area("Enter text here:")
if st.button("Predict Sentiment"):
    if user_input:
        sentiment = predict_sentiment(user_input)
        st.write(f"**Predicted Sentiment:** {sentiment}")
    else:
        st.write("⚠️ Please enter some text before predicting.")
