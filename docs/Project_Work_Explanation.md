# 📘 Complete Project Work Explanation — Layman's Guide
### Project: `AI-Powered-Task-Management-System`

---

## 📌 1. Executive Summary & Goal (High-Level Overview)

In agile software engineering teams, managing task tickets manually is slow, error-prone, and often leads to:
1. **Wrong Task Categories**: Developers mislabeling bugs, features, or UI tasks.
2. **Inaccurate Priorities**: Tickets marked with wrong urgency levels.
3. **Employee Burnout**: Assigning too many tasks to the same high-performing developers while others remain underutilized.

### What This Project Built:
An automated, data-driven AI system that:
- **Categorizes tasks automatically (92.45% accuracy)** using Natural Language Processing (NLP).
- **Predicts task priority levels (83.80% accuracy)** using Machine Learning (XGBoost & Random Forest).
- **Balances workload intelligently** using a multi-factor scoring formula to assign tickets fairly based on skill, experience, department alignment, performance, and current workload.

---

## 📁 2. Step-by-Step Breakdown of What Was Done

---

### 🔹 PART 1: Data Ingestion & Cleaning
**Notebooks**: [`01_Data_Loading_and_EDA.ipynb`](file:///c:/project/task%20management%20system/AI-Powered-Task-Management-System/notebooks/01_Data_Loading_and_EDA.ipynb) & [`02_Missing_Value_Analysis.ipynb`](file:///c:/project/task%20management%20system/AI-Powered-Task-Management-System/notebooks/02_Missing_Value_Analysis.ipynb)

- **Data Loaded**: 11 raw CSV files stored in [`data/raw/`](file:///c:/project/task%20management%20system/AI-Powered-Task-Management-System/data/raw/), including 10,000 tasks ([`tasks.csv`](file:///c:/project/task%20management%20system/AI-Powered-Task-Management-System/data/raw/tasks.csv)), 30,000 comments ([`comments.csv`](file:///c:/project/task%20management%20system/AI-Powered-Task-Management-System/data/raw/comments.csv)), employee profiles ([`employees.csv`](file:///c:/project/task%20management%20system/AI-Powered-Task-Management-System/data/raw/employees.csv)), skills, performance metrics, and workload data.
- **Cleaning Strategy**: Checked for missing values, handled null titles/descriptions, formatted timestamps, and exported clean datasets to [`data/processed/`](file:///c:/project/task%20management%20system/AI-Powered-Task-Management-System/data/processed/) (`*_clean.csv` and `*_processed.csv`).

---

### 🔹 PART 2: Exploratory Data Analysis (EDA)
**Notebook**: [`03_Exploratory_Data_Analysis.ipynb`](file:///c:/project/task%20management%20system/AI-Powered-Task-Management-System/notebooks/03_Exploratory_Data_Analysis.ipynb)

- **Statistical Analysis & Visualizations**:
  - Analyzed story point distributions, actual vs estimated hours, and department workloads.
  - Investigated task severity, ticket status progression, and developer performance metrics.
  - Exported 30+ visual analytical charts to [`outputs/charts/`](file:///c:/project/task%20management%20system/AI-Powered-Task-Management-System/outputs/charts/).

---

### 🔹 PART 3: Natural Language Processing (NLP)
**Notebooks**: [`04_NLP_Preprocessing.ipynb`](file:///c:/project/task%20management%20system/AI-Powered-Task-Management-System/notebooks/04_NLP_Preprocessing.ipynb) & [`04_NLP_Analysis.ipynb`](file:///c:/project/task%20management%20system/AI-Powered-Task-Management-System/notebooks/04_NLP_Analysis.ipynb)  
*(Detailed technical breakdown in [`docs/NLP_Analysis_Explanation.md`](file:///c:/project/task%20management%20system/AI-Powered-Task-Management-System/docs/NLP_Analysis_Explanation.md))*

1. **Text Preprocessing & TF-IDF Vectorization**:
   - Merged task titles and descriptions.
   - Cleaned text (lowercasing, stop-word removal) and vectorized using **TF-IDF** (Term Frequency-Inverse Document Frequency).
2. **Task Category Classification**:
   - Model: **Multinomial Naive Bayes (`MultinomialNB`)**.
   - Result: Achieved **92.45% Accuracy** in classifying tasks into categories (*Bug*, *Feature*, *DevOps*, *UI/UX*, *Data Engineering*, *Security*, *Testing*).
3. **Sentiment Analysis**:
   - Model: **VADER Sentiment Analyzer** on 30,000 ticket comments.
   - Result: Classified developer feedback into Positive, Neutral, and Negative sentiments.
4. **Topic Modeling (LDA)**:
   - Extracted hidden themes and key topic clusters across task descriptions.

---

### 🔹 PART 4: Machine Learning Priority Prediction
**Notebooks**: [`07_Priority_Prediction_and_Workload_Balancing.ipynb`](file:///c:/project/task%20management%20system/AI-Powered-Task-Management-System/notebooks/07_Priority_Prediction_and_Workload_Balancing.ipynb) & [`08_XGBoost_Priority_Prediction.ipynb`](file:///c:/project/task%20management%20system/AI-Powered-Task-Management-System/notebooks/08_XGBoost_Priority_Prediction.ipynb)

- **Goal**: Predict ticket priority level (*Low*, *Medium*, *High*, *Critical*).
- **Models Used**:
  - **Random Forest Classifier**: Baseline priority classification.
  - **XGBoost Classifier + GridSearchCV**: Fine-tuned hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`).
- **Performance**: Achieved **83.80% Accuracy** (F1 Score: 79.52%).
- **Top Predictive Features**: `Story_Points`, `Estimated_Hours`, `Severity`, `SLA_Hours`, `Task_Duration`.
- **Model Storage**: Trained model weights saved to [`models/random_forest_priority.pkl`](file:///c:/project/task%20management%20system/AI-Powered-Task-Management-System/models/random_forest_priority.pkl).

---

### 🔹 PART 5: Intelligent Workload Balancing Engine
**Notebook**: [`09_Intelligent_Workload_Balancing.ipynb`](file:///c:/project/task%20management%20system/AI-Powered-Task-Management-System/notebooks/09_Intelligent_Workload_Balancing.ipynb)

- **Goal**: Dynamically assign unassigned tickets to the most suitable available developer.
- **Assignment Score Formula**:
  $$S_{assignment} = 0.35 \cdot S_{skill} + 0.20 \cdot S_{dept} + 0.20 \cdot S_{perf} + 0.10 \cdot S_{exp} + 0.15 \cdot S_{util}$$
- **Key Safety Guardrails**:
  - **Overload Prevention**: Automatically skips developers with >80% current capacity utilization.
  - **Skill & Department Match**: Maximizes task-developer alignment.
- **Results Saved**: Exported complete assignment recommendations to [`outputs/workload_assignment_results.csv`](file:///c:/project/task%20management%20system/AI-Powered-Task-Management-System/outputs/workload_assignment_results.csv).
- **Visualizations Generated**:
  - [`outputs/charts/department_workload.png`](file:///c:/project/task%20management%20system/AI-Powered-Task-Management-System/outputs/charts/department_workload.png) — Task allocation across departments.
  - [`outputs/charts/employee_utilization.png`](file:///c:/project/task%20management%20system/AI-Powered-Task-Management-System/outputs/charts/employee_utilization.png) — Developer utilization post-assignment.

---

### 🔹 PART 6: Production Pipeline Scripts (`src/`)

To allow non-notebook production execution, standalone Python scripts were created in [`src/`](file:///c:/project/task%20management%20system/AI-Powered-Task-Management-System/src/):

| Script | Purpose |
| :--- | :--- |
| [`feature_extraction.py`](file:///c:/project/task%20management%20system/AI-Powered-Task-Management-System/src/feature_extraction.py) | Combines title & description, extracts TF-IDF features. |
| [`train_naive_bayes.py`](file:///c:/project/task%20management%20system/AI-Powered-Task-Management-System/src/train_naive_bayes.py) | Trains the Naive Bayes classification model. |
| [`evaluate.py`](file:///c:/project/task%20management%20system/AI-Powered-Task-Management-System/src/evaluate.py) | Evaluates accuracy, precision, recall, and classification reports. |
| [`priority_prediction.py`](file:///c:/project/task%20management%20system/AI-Powered-Task-Management-System/src/priority_prediction.py) | Runs priority model training, grid search & saves `.pkl` artifact. |
| [`run.py`](file:///c:/project/task%20management%20system/AI-Powered-Task-Management-System/src/run.py) | Execution entry point for running evaluation. |

---

## 📈 3. Summary of Results & Deliverables

- **Category Classifier**: 92.45% Accuracy
- **Priority Predictor**: 83.80% Accuracy
- **Workload Balancer**: Fully functional multi-factor assignment engine
- **Saved Results**: [`outputs/workload_assignment_results.csv`](file:///c:/project/task%20management%20system/AI-Powered-Task-Management-System/outputs/workload_assignment_results.csv)
- **Visual Charts**: 32 analytics images in [`outputs/charts/`](file:///c:/project/task%20management%20system/AI-Powered-Task-Management-System/outputs/charts/)
