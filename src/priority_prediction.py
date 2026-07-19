# ==========================================
# IMPORT LIBRARIES
# ==========================================
import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
# ==========================================
# LOAD DATASET
# ==========================================
df = pd.read_csv("data/raw/tasks.csv")

# Display first 5 rows
print(df.head())
# Selecting useful columns
features = [
    "Task_Title",
    "Task_Description",
    "Category",
    "Severity",
    "Story_Points",
    "Estimated_Hours",
    "Environment"
]

target = "Priority"

# Display selected columns
print("\nSelected Features:")
print(features)

print("\nTarget Variable:")
print(target)
from sklearn.preprocessing import LabelEncoder
# ==========================================
# LABEL ENCODING
# ==========================================
encoder = LabelEncoder()

# Convert text columns into numbers
for column in features:
    if df[column].dtype == "object":
        df[column] = encoder.fit_transform(df[column])

# Convert target column
df[target] = encoder.fit_transform(df[target])

print("\nEncoded Dataset:")
print(df[features + [target]].head())
# Create Features (X) and Target (y)
X = df[features]
y = df[target]

print("\nShape of Features:", X.shape)
print("Shape of Target:", y.shape)
# ==========================================
# TRAIN TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)
# ==========================================
# RANDOM FOREST MODEL
# ==========================================
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

print("\nRandom Forest Model Trained Successfully!")
# Predict priorities
y_pred = model.predict(X_test)

print("\nFirst 10 Predictions:")
print(y_pred[:10])
# ==========================================
# MODEL EVALUATION
# ==========================================
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n-----------------------------")
print("GridSearchCV Started...")
print("-----------------------------")
# ==========================================
# GRID SEARCH CV
# ==========================================
parameters = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=parameters,
    cv=3,
    scoring="accuracy"
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Accuracy:")
print(grid_search.best_score_)
print("\n-----------------------------")
print("Workload Balancing")
print("-----------------------------")

# ==========================================
# WORKLOAD BALANCING
# ==========================================
employee_workload = {
    "Alice": 8,
    "Bob": 4,
    "Charlie": 2,
    "David": 6
}

print("\nCurrent Workload:")

for employee, tasks in employee_workload.items():
    print(f"{employee}: {tasks} tasks")

# Find employee with minimum workload
assigned_employee = min(employee_workload, key=employee_workload.get)

print("\nNew Task Assigned To:")
print(assigned_employee)
# Save trained model
joblib.dump(model, "models/random_forest_priority.pkl")

print("\nModel saved successfully!")