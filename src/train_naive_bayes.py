from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from feature_extraction import extract_features


def train_model():

    X, y = extract_features()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = MultinomialNB()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return y_test, y_pred