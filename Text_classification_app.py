import streamlit as st
import joblib
import pandas as pd
from keras.models import load_model
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Set the tab title
st.set_page_config("Review Classification")

# Set the page title
st.title("Positive and Negative Review Classification Project")

# Set header
st.subheader("By Vaishnavi Badade")

# Load the pre-trained model and the FITTED TF-IDF vectorizer
model = load_model("TextClassification.keras")
tfidf = joblib.load("tfidf_vectorizer.pkl")  # ✅ FIX: Load the fitted vectorizer

# Create input box for user review
review = st.text_input("Review")

# Predict button
submit = st.button("Predict Sentiment")

def preprocess(text):
    text = text.lower()
    pattern = r"[^a-z ]"
    text = re.sub(pattern, "", text)
    return text

def predict(text):
    review_updated = preprocess(text)
    review_pre = tfidf.transform([review_updated]).toarray()  # Now works correctly
    probs = model.predict(review_pre, verbose=0)
    if probs > 0.5:
        st.subheader("Positive Review ✅")
    else:
        st.subheader("Negative Review ❌")

if submit:
    if review.strip() == "":
        st.warning("Please enter a review before predicting.")
    else:
        predict(review)
