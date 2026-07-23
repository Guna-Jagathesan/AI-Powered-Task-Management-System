# 🤖 AI-Powered Task Management System

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Active](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

An end-to-end Machine Learning (ML) and Natural Language Processing (NLP) framework built to automate **task prioritization**, **category classification**, and **intelligent workload balancing** for software engineering and project management teams.

---

## 📌 Project Overview & Objectives

In modern agile software teams, ticket management often suffers from manual effort, improper priority tagging, and uneven workload distribution. 

This repository provides an automated, data-driven system to:
1. **Categorize Tasks Automatically (NLP)**: Classifies tasks (e.g., *Bug*, *Feature*, *DevOps*, *UI/UX*, *Data Engineering*, *Security*) based on task titles and descriptions using **TF-IDF Vectorization** and **Multinomial Naive Bayes** (**92.45% Accuracy**).
2. **Predict Task Priority (ML)**: Predicts task urgency (*Low*, *Medium*, *High*, *Critical*) using **Random Forest** and **XGBoost Classifier** (**83.80% Accuracy**).
3. **Balance Team Workload (Intelligent Allocation)**: Dynamically assigns unassigned tickets to team members by calculating a multi-factor **Assignment Score** based on skill match, department alignment, experience, performance history, and capacity utilization.
4. **Data Cleaning & EDA**: Automated processing pipelines converting raw tracking data into clean, structured analytics datasets.

---

## 📁 Repository & File Structure

```
AI-Powered-Task-Management-System/
├── app/                              # Production interface wrappers & service components
├── data/                             # Data storage
│   ├── raw/                          # Raw datasets (tasks, employees, workload, sprints, etc.)
│   └── processed/                    # Cleaned & transformed datasets (*_clean.csv, *_nlp.csv)
├── docs/                             # Project documentation
│   ├── Project_Work_Explanation.md   # Complete step-by-step layman explanation guide
│   └── NLP_Analysis_Explanation.md   # Deep-dive documentation into the NLP methodology
├── images/                           # System diagrams and assets
├── models/                           # Serialized machine learning models (.pkl)
│   └── random_forest_priority.pkl    # Saved Random Forest priority prediction model
├── notebooks/                        # Step-by-step Jupyter Notebooks for analysis & modeling
│   ├── 01_Data_Loading_and_EDA.ipynb                       # Initial data inspection
│   ├── 02_Missing_Value_Analysis.ipynb                     # Data cleaning & imputation
│   ├── 03_Exploratory_Data_Analysis.ipynb                  # EDA & feature distributions
│   ├── 04_NLP_Analysis.ipynb                               # Text exploration & sentiment
│   ├── 04_NLP_Preprocessing.ipynb                          # TF-IDF & text cleaning
│   ├── 07_Priority_Prediction_and_Workload_Balancing.ipynb # Initial priority baseline
│   ├── 08_XGBoost_Priority_Prediction.ipynb                # XGBoost priority modeling
│   └── 09_Intelligent_Workload_Balancing.ipynb             # Multi-factor workload engine
├── outputs/                          # Generated predictions & exported reports
│   ├── task_assignment_results.csv                         # Final task-employee assignments
│   ├── workload_assignment_results.csv                     # Workload allocation log
│   └── charts/                       # 30+ exported analytical visualizations
├── src/                              # Production Python scripts
│   ├── feature_extraction.py         # TF-IDF text feature extraction module
│   ├── train_naive_bayes.py          # Naive Bayes training script
│   ├── evaluate.py                   # Model evaluation suite (Accuracy, F1, Precision)
│   ├── priority_prediction.py        # Random Forest model + GridSearchCV + Allocation
│   └── run.py                        # Pipeline entry point
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 📊 Key Results & Model Performance

### 1. NLP Task Category Classification (`MultinomialNB`)
* **Accuracy**: `92.45%`
* **Weighted Precision**: `92.35%`
* **Weighted Recall**: `92.45%`
* **Best Performing Categories**: Data Engineering (F1: 0.99), Bug (F1: 0.98), UI/UX (F1: 0.97), Testing (F1: 0.95), Feature (F1: 0.95).

### 2. Priority Prediction (`XGBoost` & `RandomForest`)
* **Accuracy**: `83.80%`
* **F1 Score**: `79.52%`
* **Key Features**: `Story_Points`, `Estimated_Hours`, `Severity`, `SLA_Hours`, `Task_Duration`.

### 3. Intelligent Workload Balancing Engine
Computes an **Assignment Score** ($S_{assignment}$) for eligible, available, non-overloaded team members:

$$S_{assignment} = 0.35 \cdot S_{skill} + 0.20 \cdot S_{dept} + 0.20 \cdot S_{perf} + 0.10 \cdot S_{exp} + 0.15 \cdot S_{util}$$

* **Key Benefits**:
  - Prevents employee burnout by skipping overloaded personnel ($>80\%$ utilization).
  - Maximizes skill and department alignment.
  - Distributes tasks evenly across departments.

---

## 🚀 Quickstart for Collaborators

### 1. Clone & Setup Environment
```bash
# Clone the repository
git clone https://github.com/your-org/AI-Powered-Task-Management-System.git
cd AI-Powered-Task-Management-System

# Create virtual environment (optional but recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Python Scripts
* **Execute Category Classification Pipeline**:
  ```bash
  python src/run.py
  ```
* **Train Priority Prediction & Run Workload Allocator**:
  ```bash
  python src/priority_prediction.py
  ```

### 3. Explore Jupyter Notebooks
Launch Jupyter Notebook to view interactive experiments, visualizations, and step-by-step model training:
```bash
jupyter notebook
```
Navigate to `notebooks/09_Intelligent_Workload_Balancing.ipynb` or `notebooks/08_XGBoost_Priority_Prediction.ipynb`.

---

## 📈 Visualizations Overview

All key charts are exported to `outputs/charts/`, including:
* [`department_workload.png`](file:///c:/project/task%20management%20system/AI-Powered-Task-Management-System/outputs/charts/department_workload.png) — Department-wise Task Allocation.
* [`employee_utilization.png`](file:///c:/project/task%20management%20system/AI-Powered-Task-Management-System/outputs/charts/employee_utilization.png) — Post-assignment Team Utilization Distribution.
* [`classification_confusion_matrix.png`](file:///c:/project/task%20management%20system/AI-Powered-Task-Management-System/outputs/charts/classification_confusion_matrix.png) — Confusion matrix for Category Classification.
* [`xgboost_feature_importance.png`](file:///c:/project/task%20management%20system/AI-Powered-Task-Management-System/outputs/charts/xgboost_feature_importance.png) — Feature importance plot for priority modeling.

---

## 🤝 Contribution Guidelines

We welcome contributions! When adding features or refactoring:
1. Ensure new dependencies are added to `requirements.txt`.
2. Save raw data in `data/raw/` and processed datasets in `data/processed/`.
3. Save trained models as `.pkl` artifacts in `models/`.
4. Run `python src/run.py` to verify that existing pipelines pass without errors.

---

## 📄 License
This project is licensed under the **MIT License**.
