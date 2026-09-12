# Fake News Detection using NLP + Machine Learning

A machine learning project that classifies news articles as **Real** or **Fake**
using TF-IDF text vectorization and classic ML models (Logistic Regression,
Naive Bayes), with a live Streamlit demo app.

---

## 1. Setup (Day 1 — Morning, ~30 min)

### 1.1 Install Python
Make sure you have Python 3.9+ installed. Check with:
```bash
python --version
```

### 1.2 Create a project folder and virtual environment
```bash
mkdir fake-news-detector
cd fake-news-detector
python -m venv venv

# Activate it:
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 1.3 Copy in the project files
Place `train_model.py`, `app.py`, and `requirements.txt` into this folder
(you already have them if you downloaded this project bundle).

### 1.4 Install dependencies
```bash
pip install -r requirements.txt
```

---

## 2. Get the Dataset (Day 1 — Morning)

1. Go to Kaggle: **"Fake and Real News Dataset"** by Clément Bisaillon
   https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
2. Download `Fake.csv` and `True.csv`
3. Create a `data/` folder in your project and place both files inside:
   ```
   fake-news-detector/
   ├── data/
   │   ├── Fake.csv
   │   └── True.csv
   ├── model/
   ├── train_model.py
   ├── app.py
   └── requirements.txt
   ```

If you don't have a Kaggle account, sign up free at kaggle.com — it takes
2 minutes and you'll need it eventually for ML work anyway.

---

## 3. Train the Model (Day 1 — Afternoon/Evening)

Run:
```bash
python train_model.py
```

This will:
- Load and label both CSVs (0 = Fake, 1 = Real)
- Clean the text (lowercase, remove URLs/numbers/punctuation)
- Convert text to TF-IDF features
- Train Logistic Regression and Naive Bayes
- Print accuracy, precision, recall, F1 for both
- Save a confusion matrix plot to `model/confusion_matrix.png`
- Save the best-performing model to `model/fake_news_model.pkl`
- Save the vectorizer to `model/vectorizer.pkl`

**Expected result:** Logistic Regression typically hits **98–99% accuracy**
on this dataset (it's a relatively "easy" dataset because fake and real
articles differ noticeably in writing style). Don't be surprised — this is
a good, presentable number, just be ready to explain the "why" (see Section 6).

---

## 4. Run the Demo App (Day 2 — Morning)

Once training is done and `model/fake_news_model.pkl` exists, run:
```bash
streamlit run app.py
```

This opens a browser window where you can:
- Paste in any news headline or article
- Click "Check News"
- See a Real/Fake prediction with a confidence percentage

Try it with a few real headlines from a legitimate news site and a few
obviously fabricated ones to build demo examples ahead of time.

---

## 5. Push to GitHub (Day 2 — Afternoon)

```bash
git init
git add .
git commit -m "Fake news detection project using NLP and ML"
git branch -M main
git remote add origin https://github.com/<your-username>/fake-news-detector.git
git push -u origin main
```

**Important:** Add a `.gitignore` with:
```
venv/
__pycache__/
data/
```
(Don't commit the raw dataset — it's large and easily re-downloadable;
committing it also risks license/redistribution issues.)

---

## 6. Presentation Talking Points (Day 2 — Evening)

Be ready to explain, in your own words:

- **The problem**: misinformation spreads fast online; automated detection
  helps flag suspicious content for review.
- **The pipeline**: raw text → cleaning → TF-IDF vectorization → classifier
  → prediction.
- **Why TF-IDF**: it weighs words by how distinctive they are to a document
  relative to the whole corpus, rather than just counting raw frequency.
- **Why Logistic Regression works well here**: it's a strong baseline for
  high-dimensional sparse text features, and it's interpretable (you can
  inspect top weighted words for each class if asked).
- **Limitations** (mention these proactively — it shows maturity):
  - Trained on articles from a specific time/source, so it may not
    generalize to today's writing styles or new topics.
  - It detects *writing style patterns*, not *factual accuracy* — a
    well-written fabricated story could fool it, and a poorly written
    true story could also be misclassified.
  - Real-world fake news detection needs source credibility, cross-referencing,
    and human fact-checkers alongside any ML model.
- **Possible extensions** if asked "what would you do with more time":
  transformer-based models (BERT/DistilBERT), multilingual support,
  or a browser extension front-end.

---

## 7. Resume Bullet Point

> Built and deployed a Fake News Detection system using NLP (TF-IDF) and
> Logistic Regression, achieving ~98% accuracy on a 44K-article dataset;
> created an interactive Streamlit demo for real-time predictions.

---

## Project Structure
```
fake-news-detector/
├── data/                     # Fake.csv, True.csv (not committed to git)
├── model/
│   ├── fake_news_model.pkl   # trained model (generated)
│   ├── vectorizer.pkl        # TF-IDF vectorizer (generated)
│   └── confusion_matrix.png  # evaluation plot (generated)
├── train_model.py            # training script
├── app.py                    # Streamlit demo app
├── requirements.txt
└── README.md
```
