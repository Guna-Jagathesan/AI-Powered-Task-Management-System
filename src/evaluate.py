from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report
)

from train_naive_bayes import train_model


def evaluate_model():

    y_test, y_pred = train_model()

    print("\n===== MODEL PERFORMANCE =====\n")

    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred, average='weighted'):.4f}")

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))