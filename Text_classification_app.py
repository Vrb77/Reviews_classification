import re
import numpy as np
import streamlit as st
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from keras.models import load_model

# ── These classes MUST be defined here so joblib can unpickle the pipeline ──

class TextCleaner(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._clean(text) for text in X]

    def _clean(self, text):
        text = text.lower()
        pattern = r"[^a-z ]"
        text = re.sub(pattern, "", text)
        return text


class TextTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.tfidf = TfidfVectorizer()
        self.tfidf.fit(X)
        return self

    def transform(self, X):
        return self.tfidf.transform(X).toarray()

# ────────────────────────────────────────────────────────────────────────────

# Set the tab title
st.set_page_config(page_title="Review Classification")

# Set the page title
st.title("Positive and Negative Review Classification Project")

# Set header
st.subheader("By Vaishnavi Badade")

# Load the pipeline (data cleaning, preprocessing) and model
pre = joblib.load("text_classification_pre.joblib")
model = load_model("TextClassification.keras")

# Text input (reviews are text, not numbers)
review = st.text_input("Enter your review")

# Predict button
submit = st.button("Predict Sentiment")

if submit:
    if not review.strip():
        st.warning("Please enter a review before predicting.")
    else:
        # Pipeline expects a list of strings
        transformed_text = pre.transform([review])

        # Prediction
        preds = model.predict(transformed_text)
        if preds[0][0] > 0.5:
            st.subheader("✅ Positive Review")
        else:
            st.subheader("❌ Negative Review")
