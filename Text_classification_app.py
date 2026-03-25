import streamlit as st
import joblib
import pandas as pd
from keras.models import load_model
import re
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf =TfidfVectorizer()

# set the tab title
st.set_page_config("Review Classification")

# Set the page title
st.title("Positive and Negative Review Classification Project")

# Set header
st.subheader("By Vaishnavi Badade")

model = load_model("TextClassification.keras")

# Create Input boxes that takes input from the user 
review = st.text_input("review")

# Include a button. After providing all the inputs, user will click on the button. The button should provide the necessary predictions
submit = st.button("Predict Sentiment")

def preprocess(text):
  text=text.lower()
  pattern = r"[^a-z ]"
  text=re.sub(pattern,"",text)
  return text

def predict(text):
  review_updated = preprocess(text)
  review_pre = tfidf.transform([review_updated]).toarray()
  probs = model.predict(review_pre,verbose=0)
  if probs>0.5:
    st.subheader("Positive Review")
  else:
    st.subheader("Negative Review")

if submit:
    predict(review)  
