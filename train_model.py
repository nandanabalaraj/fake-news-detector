"""
train_model.py
Fake News Detection - Training Script

Steps:
1. Load and label the Fake/True news datasets
2. Clean and preprocess text
3. Convert text to TF-IDF features
4. Train Logistic Regression + Naive Bayes, compare
5. Evaluate with accuracy, precision, recall, F1, confusion matrix
6. Save the best model + vectorizer for later use in app.py

Dataset: "Fake and Real News Dataset" by Clément Bisaillon on Kaggle
Link: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
Download Fake.csv and True.csv and place them in the data/ folder before running this.
"""

import re
import string
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------
# STEP 1: Load data
# ---------------------------------------------------------
print("Loading data...")
fake_df = pd.read_csv("data/Fake.csv")
true_df = pd.read_csv("data/True.csv")

fake_df["label"] = 0  # 0 = Fake
true_df["label"] = 1  # 1 = Real

df = pd.concat([fake_df, true_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle rows

print(f"Total articles: {len(df)}")
print(df["label"].value_counts())

# Combine title + text for a richer signal
df["content"] = df["title"].astype(str) + " " + df["text"].astype(str)

# ---------------------------------------------------------
# STEP 2: Clean text
# ---------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"\d+", "", text)                       # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()               # remove extra whitespace
    return text

print("Cleaning text...")
df["clean_content"] = df["content"].apply(clean_text)

# ---------------------------------------------------------
# STEP 3: Train/test split + TF-IDF
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

print("Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ---------------------------------------------------------
# STEP 4: Train models
# ---------------------------------------------------------
print("Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

print("Training Naive Bayes...")
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)


def evaluate(model, name):
    preds = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print(f"\n--- {name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(classification_report(y_test, preds, target_names=["Fake", "Real"]))
    return acc, preds


lr_acc, lr_preds = evaluate(lr_model, "Logistic Regression")
nb_acc, nb_preds = evaluate(nb_model, "Naive Bayes")

# ---------------------------------------------------------
# STEP 5: Pick the better model, save confusion matrix plot
# ---------------------------------------------------------
if lr_acc >= nb_acc:
    best_model, best_preds, best_name = lr_model, lr_preds, "Logistic Regression"
else:
    best_model, best_preds, best_name = nb_model, nb_preds, "Naive Bayes"

print(f"\nBest model: {best_name}")

cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("model/confusion_matrix.png")
print("Saved confusion matrix plot to model/confusion_matrix.png")

# ---------------------------------------------------------
# STEP 6: Save model + vectorizer
# ---------------------------------------------------------
joblib.dump(best_model, "model/fake_news_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")
print("\nSaved model to model/fake_news_model.pkl")
print("Saved vectorizer to model/vectorizer.pkl")
print("\nDone! You can now run: streamlit run app.py")
