import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords if they aren't already on the system running the app
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# --- 1. Load the Models ---
@st.cache_resource # This keeps the models in memory so the app runs faster
def load_models():
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return vectorizer, model

vectorizer, model = load_models()

# --- 2. Setup the Text Cleaning Pipeline ---
stop_words = set(stopwords.words('english'))
stop_words.discard("not")
stop_words.discard("no")
stop_words.discard("nor")
stemmer = PorterStemmer()

def clean_text(text):
    # HTML, lowercase, special characters
    text = re.sub('<.*?>', '', text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Stopwords and Stemming
    words = text.split()
    processed_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(processed_words)

# --- 3. Build the Streamlit User Interface ---
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Type a review below to see if our Machine Learning model thinks it is Positive or Negative!")

# Create a text input box for the user
user_input = st.text_area("Enter your review here:", height=150)

# Create a button to trigger the prediction
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review first!")
    else:
        # 1. Clean the user's text
        cleaned_input = clean_text(user_input)
        
        # 2. Vectorize the text
        vectorized_input = vectorizer.transform([cleaned_input])
        
        # 3. Make Prediction
        prediction = model.predict(vectorized_input)[0]
        confidence = model.predict_proba(vectorized_input).max()
        
        # 4. Display the results beautifully
        if prediction == 1:
            st.success(f"**Positive!** 🟢 (Confidence: {confidence*100:.1f}%)")
            st.balloons() # Fun little Streamlit animation
        else:
            st.error(f"**Negative.** 🔴 (Confidence: {confidence*100:.1f}%)")