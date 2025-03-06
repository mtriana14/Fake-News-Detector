import streamlit as st
import os
import subprocess
import pickle
import spacy
import requests
from sklearn.feature_extraction.text import TfidfVectorizer

#  Google Fact Check API Key 
FACT_CHECK_API_KEY = "AIzaSyC9rNcQVanALN5W4SZwBYu0eVgc1wQtTNA"

#  Load the saved model and vectorizer with a loading spinner
st.set_page_config(page_title="Fake News Detector", page_icon="", layout="wide")

with st.spinner("Loading AI model... Please wait..."):
    MODEL_FILE = "fake_news_model.pkl"
    VECTORIZER_FILE = "tfidf_vectorizer.pkl"
    with open(MODEL_FILE, "rb") as model_file:
        model = pickle.load(model_file)
    with open(VECTORIZER_FILE, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

st.success(" AI Model Loaded Successfully!")

#  Lazy Load NLP Model (Caching)
@st.cache_resource

def load_nlp():
    """Ensure spaCy model is available before loading"""
    model_name = "en_core_web_sm"
    try:
        return spacy.load(model_name, disable=["parser", "ner"])
    except OSError:  # Model not found, install it
        subprocess.run(["python", "-m", "spacy", "download", model_name])
        return spacy.load(model_name, disable=["parser", "ner"])


nlp = load_nlp()

#  Function to clean text using spaCy
def clean_text(text):
    """Cleans and preprocesses text using spaCy"""
    doc = nlp(text.lower())
    words = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(words)

#  Function to check statement with Google Fact Check API
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
            return f" **Fact Check Result**: **{rating}** (Source: **{source}**)\n🔗 [Read More]({url})"
    return " No fact-checking results found for this statement."

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
    classification = "**REAL NEWS**" if prediction == 1 else " **FAKE NEWS**"

    return classification, fact_check_result

#   Streamlit UI 
st.markdown("<h1 style='text-align: center; color: #FF5733;'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Enter a news article or statement to check if it's real or fake.</h4>", unsafe_allow_html=True)

st.write("---")

#  User Input Section
user_input = st.text_area(" Paste a news article or statement here:", height=150)

#  Prediction Button
if st.button(" Check News"):
    if user_input.strip():
        classification, fact_check_result = predict_news(user_input)

        #  Display the results
        st.markdown(f"<h3 style='color: #007BFF;'> Classification Result:</h3>", unsafe_allow_html=True)
        st.markdown(f"<div style='border-radius:10px; padding:10px; background-color:#F4F4F4;'>"
                    f"<h2>{classification}</h2>"
                    f"</div>", unsafe_allow_html=True)

        # Display Fact Check Results
        st.markdown(f"<h3 style='color: #28A745;'>🔎 Fact Check Verification:</h3>", unsafe_allow_html=True)

        #  Make Fact Check link clickable
        if "Read More" in fact_check_result:
            st.success(fact_check_result)
        else:
            st.warning(fact_check_result)

    else:
        st.warning(" Please enter a news article or statement to analyze.")

#  Footer
st.write("---")
st.markdown("<h6 style='text-align: center; color: grey;'>Powered by Machine Learning & Google Fact Check API</h6>", unsafe_allow_html=True)

#  "Done by" Section with LinkedIn Button
st.markdown("<h4 style='text-align: center;'>👨‍💻 Done by <a href='https://www.linkedin.com/in/melanitriana' target='_blank'>MT</a></h4>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="text-align: center;">
        <a href="https://www.linkedin.com/in/melanitriana" target="_blank">
            <button style="padding:10px 20px; font-size:16px; background-color:#0A66C2; color:white; border:none; border-radius:5px; cursor:pointer;">
                🔗 LinkedIn
            </button>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)
