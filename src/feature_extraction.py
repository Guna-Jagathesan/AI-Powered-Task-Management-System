import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_features():
    # Load dataset
    df = pd.read_csv("data/raw/tasks.csv")

    # Combine title and description
    df["Text"] = (
        df["Task_Title"].fillna("") + " " +
        df["Task_Description"].fillna("")
    )

    # Features and labels
    X = df["Text"]
    y = df["Category"]

    # TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english")
    X_tfidf = vectorizer.fit_transform(X)

    return X_tfidf, y