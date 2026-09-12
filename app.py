"""
app.py
Fake News Detection - Streamlit Demo App

Run with: streamlit run app.py
Loads the trained model + vectorizer and lets the user paste in a
news headline/article to get a Real vs Fake prediction with confidence.
"""

import re
import string
import joblib
import streamlit as st

# ---------------------------------------------------------
# Load model + vectorizer (cached so it only loads once)
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model/fake_news_model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    return model, vectorizer


model, vectorizer = load_artifacts()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.write(
    "Paste a news headline or article below and the model will predict "
    "whether it's likely **Real** or **Fake**, using a TF-IDF + Machine "
    "Learning model trained on the Fake and Real News dataset."
)

user_input = st.text_area("Enter news text here:", height=200)

if st.button("Check News"):
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]

        label = "🟢 Real" if prediction == 1 else "🔴 Fake"
        confidence = proba[prediction] * 100

        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: **{confidence:.2f}%**")

        st.progress(int(confidence))

st.markdown("---")
st.caption(
    "Model: TF-IDF + Logistic Regression / Naive Bayes | "
    "Dataset: Fake and Real News Dataset (Kaggle)"
)
