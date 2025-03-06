import streamlit as st
import os
import pickle
import spacy
import subprocess
import requests
from sklearn.feature_extraction.text import TfidfVectorizer


FACT_CHECK_API_KEY = st.secrets["api"]["FACT_CHECK_API_KEY"]


# Load the saved model and vectorizer
MODEL_FILE = "fake_news_model.pkl"
VECTORIZER_FILE = "tfidf_vectorizer.pkl"




nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def install_spacy_model():
    """Ensure spaCy model is installed before using it."""
    model_name = "en_core_web_sm"
    
    try:
        # Try loading the model first
        spacy.load(model_name)
    except OSError:
        # If model is missing, attempt to download it
        try:
            subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
        except subprocess.CalledProcessError:
            print(f"Error: Could not install {model_name}. Please install it manually.")



def load_spacy_model():
    """Ensure spaCy model is available before loading"""
    try:
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError:
        print("Error: spaCy model 'en_core_web_sm' is missing.")
        print("Please install it manually using: python -m spacy download en_core_web_sm")
        return None  # Prevent crashing

nlp = load_spacy_model()
if nlp is None:
    raise SystemExit("Critical Error: Failed to load 'en_core_web_sm'. Ensure it is installed.")


# Load the trained model and vectorizer
with open(MODEL_FILE, "rb") as model_file:
    model = pickle.load(model_file)
with open(VECTORIZER_FILE, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Function to clean text using spaCy
def clean_text(text):
    """Cleans and preprocesses text using spaCy"""
    if not text or not isinstance(text, str):
        return ""
    doc = nlp(text.lower())
    words = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(words)

# Function to check statement with Google Fact Check API
def check_fact_with_api(statement):
    """Checks if a statement has been fact-checked by Google Fact Check API"""
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={statement}&key={FACT_CHECK_API_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if "claims" in data and data["claims"]:
            fact_result = data["claims"][0]  # Take the first result
            source = fact_result["claimReview"][0]["publisher"]["name"]
            rating = fact_result["claimReview"][0]["textualRating"]
            url = fact_result["claimReview"][0]["url"]
            return f"Fact Check Result: {rating} (Source: {source})\n[Read More]({url})"
    return "No fact-checking results found for this statement."

# Function to predict if news is real or fake
def predict_news(news_article):
    """Predicts if a given news article is real or fake"""
    clean_input = clean_text(news_article)
    
    # Check fact with API first
    fact_check_result = check_fact_with_api(news_article)
    
    # Convert text to TF-IDF format
    input_vector = vectorizer.transform([clean_input])
    
    # Predict using trained model
    prediction = model.predict(input_vector)[0]
    classification = "REAL NEWS" if prediction == 1 else "FAKE NEWS"

    return classification, fact_check_result

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="wide")

# Header
st.markdown("<h1 style='text-align: center; color: #333;'>Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Verify the credibility of news articles and statements.</h4>", unsafe_allow_html=True)

st.write("---")

# User Input Section
user_input = st.text_area("Enter a news article or statement here:", height=150)

# Prediction Button
if st.button("Check News"):
    if user_input.strip():
        classification, fact_check_result = predict_news(user_input)

        # Display the results
        st.markdown(f"<h3 style='color: #007BFF;'>Classification Result:</h3>", unsafe_allow_html=True)
        st.markdown(f"<div style='border-radius:10px; padding:10px; background-color:#F4F4F4;'>"
                    f"<h2>{classification}</h2>"
                    f"</div>", unsafe_allow_html=True)

        # Display Fact Check Results
        st.markdown(f"<h3 style='color: #28A745;'>Fact Check Verification:</h3>", unsafe_allow_html=True)
        st.markdown(f"<div style='border-radius:10px; padding:10px; background-color:#FFF3CD; color:#856404;'>"
                    f"<p>{fact_check_result}</p>"
                    f"</div>", unsafe_allow_html=True)

    else:
        st.warning("Please enter a news article or statement to analyze.")

# Footer
st.write("---")
st.markdown("<h6 style='text-align: center; color: grey;'>Powered by Machine Learning & Google Fact Check API</h6>", unsafe_allow_html=True)
