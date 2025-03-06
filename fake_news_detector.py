import spacy
import pickle 
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the English NLP model (disable unnecessary components for speed)
print("Loading spaCy model...")
start_time = time.time()
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
print(f"spaCy model loaded in {time.time() - start_time:.2f} seconds.")

def clean_texts(texts):
    """Function to clean a batch of texts using spaCy (Optimized for speed)"""
    cleaned_texts = []
    print("Starting text preprocessing...")
    start_time = time.time()
    for i, doc in enumerate(nlp.pipe(texts, batch_size=1000)):  # Process texts in batches
        words = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        cleaned_texts.append(" ".join(words))
        if i % 500 == 0:  # Print progress every 500 texts
            print(f"Processed {i+1}/{len(texts)} rows... Time elapsed: {time.time() - start_time:.2f} seconds")
    print(f"Text preprocessing completed in {time.time() - start_time:.2f} seconds.")
    return cleaned_texts

# Load the datasets
print("Loading dataset...")
start_time = time.time()
true_df = pd.read_csv("True.csv")  # Real news
fake_df = pd.read_csv("Fake.csv")  # Fake news
print(f"Dataset loaded in {time.time() - start_time:.2f} seconds.")

# Add labels: 1 for real, 0 for fake
true_df["label"] = 1
fake_df["label"] = 0

# Combine both datasets
df = pd.concat([true_df, fake_df], axis=0)
print(f"Total dataset size before sampling: {df.shape[0]} rows.")

# Reduce dataset size for faster debugging
df = df.sample(n=2000, random_state=42).reset_index(drop=True)
print(f"Dataset reduced to {df.shape[0]} rows.")

# Merge title and text columns
df["combined_text"] = df["title"] + " " + df["text"]

# Apply optimized batch text processing
print("Cleaning text data...")
start_time = time.time()
df["clean_text"] = clean_texts(df["combined_text"])
print(f"Text cleaning completed in {time.time() - start_time:.2f} seconds.")

# Convert text data to numerical format using TF-IDF
print("Converting text to TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=5000)  # Limit features to speed up training
X = vectorizer.fit_transform(df["clean_text"])

# Extract labels
y = df["label"]

# Split data into training and testing sets
print("Splitting dataset into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# Train Logistic Regression Model
print("Training Logistic Regression model...")
start_time = time.time()
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)
print(f"Model trained in {time.time() - start_time:.2f} seconds.")

# Evaluate the Model
print("Evaluating the model...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- SAVE TRAINED MODEL & VECTORIZER ---
print("\n💾 Saving model and vectorizer...")
with open("fake_news_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("tfidf_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("✅ Model and vectorizer saved successfully!")

# --- ADD CUSTOM NEWS PREDICTION ---
def predict_news():
    """Allow user to enter a news article for prediction"""
    print("\nEnter a news article to check if it's real or fake.")
    user_input = input("Enter your news content: ")

    # Preprocess the input text
    clean_input = clean_texts([user_input])

    # Convert text to TF-IDF format
    input_vector = vectorizer.transform(clean_input)

    # Predict using trained model
    prediction = model.predict(input_vector)[0]

    # Output result
    if prediction == 1:
        print("\n🟢 This news is classified as **REAL**.")
    else:
        print("\n🔴 This news is classified as **FAKE**.")

# Run the prediction function
predict_news()
