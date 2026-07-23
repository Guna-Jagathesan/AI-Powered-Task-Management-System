# NLP Analysis — Code Explanation
### File: `notebooks/04_NLP_Analysis.ipynb`

---

## Overview

Is notebook mein **3 NLP tasks** kiye gaye hain Task Management System ke data par:

| Part | Task | Data Used |
|------|------|-----------|
| 1 | Sentiment Analysis | `comments.csv` (30,000 rows) |
| 2 | Text Classification | `tasks.csv` (10,000 rows) |
| 3 | Topic Modeling | `tasks.csv` (10,000 rows) |

---

## Setup (Cell 1 & 2)

```python
!pip install pandas matplotlib seaborn scikit-learn nltk vaderSentiment wordcloud
```

**Kya karta hai:**  
Saari required libraries install karta hai. Ye ek baar run karna hota hai.

**Libraries ka kaam:**
- `pandas` — data load aur manipulate karna
- `matplotlib` / `seaborn` — charts banana
- `scikit-learn` — ML models (Logistic Regression, LDA, TF-IDF)
- `nltk` — stopwords (common words jaise "the", "is" remove karne ke liye)
- `vaderSentiment` — sentiment analysis tool
- `wordcloud` — word cloud images banana

---

## Cell 3 — Data Load

```python
tasks = pd.read_csv('../data/raw/tasks.csv')
comments = pd.read_csv('../data/raw/comments.csv')
```

**Kya karta hai:**  
Dono CSV files load karta hai.

**Output:**
```
tasks shape: (10000, 20)    → 10,000 tasks, 20 columns
comments shape: (30000, 10) → 30,000 comments, 10 columns
```

---

---

# PART 1 — Sentiment Analysis on Comments

**Goal:** Comments ka sentiment predict karna (Positive / Negative / Neutral) aur original labels se compare karna.

---

## Cell 4 — VADER Analyzer Setup

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
print(comments['Comment_Text'].value_counts().head(10))
```

**Kya karta hai:**  
VADER (Valence Aware Dictionary and sEntiment Reasoner) tool initialize karta hai. Ye ek rule-based sentiment analyzer hai jo bina training ke kaam karta hai.

**Top 10 most repeated comments dikhata hai:**
```
Customer confirmed the fix.     1565
Issue reproduced successfully.  1552
...
```
Ye dikhata hai ki comments repetitive hain — sirf ~10 unique phrases hain jo baar baar repeat hote hain.

---

## Cell 5 — Sentiment Prediction Function

```python
def get_sentiment(text):
    score = analyzer.polarity_scores(str(text))['compound']
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

comments['vader_sentiment'] = comments['Comment_Text'].apply(get_sentiment)
```

**Kya karta hai:**

1. `analyzer.polarity_scores(text)` — VADER ek dictionary return karta hai:
   - `pos`, `neg`, `neu` — individual scores
   - `compound` — overall score (-1 to +1)

2. **Threshold rules:**
   - compound >= 0.05 → **Positive**
   - compound <= -0.05 → **Negative**
   - baaki → **Neutral**

3. Har comment par ye function apply hota hai aur result `vader_sentiment` column mein save hota hai.

**Example:**
```
"Testing completed successfully." → compound = 0.4 → Positive ✓
"Unable to reproduce the issue." → compound = 0.0 → Neutral
```

---

## Cell 6 — Accuracy Check

```python
from sklearn.metrics import classification_report
print(classification_report(comments['Sentiment'], comments['vader_sentiment']))
```

**Kya karta hai:**  
Original labels (`Sentiment` column) aur VADER predictions (`vader_sentiment`) compare karta hai.

**Output:**
```
              precision  recall  f1-score
Negative       0.34      0.05      0.09
Neutral        0.34      0.60      0.43
Positive       0.33      0.35      0.34
accuracy                           0.34
```

**Kyun accuracy sirf 34% hai?**  
Kyunki comments technical hain jaise "Database migration completed." — VADER ko ye neutral lagta hai lekin original label Positive ya Negative ho sakta hai. VADER social media text ke liye design hua hai, technical text ke liye nahi.

---

## Cell 7 — Sentiment Distribution Chart

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
comments['Sentiment'].value_counts().plot(kind='bar', ax=axes[0])
comments['vader_sentiment'].value_counts().plot(kind='bar', ax=axes[1])
plt.savefig('../outputs/charts/sentiment_comparison.png')
```

**Kya karta hai:**  
Side-by-side bar charts banata hai:
- Left: Original labels ka distribution
- Right: VADER predictions ka distribution

**Saved:** `outputs/charts/sentiment_comparison.png`

---

## Cell 8 — Word Clouds per Sentiment

```python
for sentiment in ['Positive', 'Neutral', 'Negative']:
    text = ' '.join(comments[comments['Sentiment'] == sentiment]['Comment_Text'])
    wc = WordCloud(...).generate(text)
    axes[i].imshow(wc)
plt.savefig('../outputs/charts/sentiment_wordclouds.png')
```

**Kya karta hai:**  
Har sentiment ke liye ek word cloud banata hai — jitna bada word, utna zyada use hua.

**Saved:** `outputs/charts/sentiment_wordclouds.png`

---

---

# PART 2 — Text Classification (Task Category Prediction)

**Goal:** Task ka title + description dekh ke automatically uski category predict karna.

---

## Cell 9 — Data Preparation

```python
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

nltk.download('stopwords')

# title aur description combine karo
tasks['text'] = tasks['Task_Title'].astype(str) + ' ' + tasks['Task_Description'].astype(str)
print('Categories:', tasks['Category'].value_counts())
```

**Kya karta hai:**

1. NLTK stopwords download karta hai (English ke common words)
2. `Task_Title` aur `Task_Description` ko ek `text` column mein merge karta hai
3. Saari categories aur unka count dikhata hai

**Kyun title + description combine kiya?**  
Zyada text = zyada information = better prediction.

---

## Cell 10 — Train/Test Split

```python
X = tasks['text']      # input (text)
y = tasks['Category']  # output (category label)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Kya karta hai:**  
Data ko 2 parts mein split karta hai:
- **80% Training** (8,000 tasks) — model seekhta hai
- **20% Testing** (2,000 tasks) — model evaluate hota hai

`random_state=42` — reproducibility ke liye (har baar same split)

---

## Cell 11 — TF-IDF Vectorization

```python
tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
```

**Kya karta hai:**  
Text ko numbers mein convert karta hai (ML models text nahi samajhte, sirf numbers).

**TF-IDF kya hai?**
- **TF (Term Frequency):** Ek word kitni baar ek document mein aaya
- **IDF (Inverse Document Frequency):** Ek word kitne documents mein aaya (common words ko kam weight)
- Result: Important words ko high score, common words ("the", "is") ko low score

**Parameters:**
- `max_features=5000` — sirf top 5000 words use karo
- `stop_words='english'` — English stopwords remove karo
- `ngram_range=(1, 2)` — single words aur 2-word phrases dono use karo (e.g., "bug fix", "code review")

**Output shape:** `(8000, 5000)` — 8000 tasks, 5000 features

---

## Cell 12 — Logistic Regression Model

```python
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))
```

**Kya karta hai:**

1. **Logistic Regression** train karta hai — ye ek simple lekin effective classification algorithm hai
2. Test data par predictions karta hai
3. Accuracy report print karta hai

**Kyun Logistic Regression?**  
Text classification ke liye fast, interpretable, aur generally achha perform karta hai.

---

## Cell 13 — Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_)
plt.savefig('../outputs/charts/classification_confusion_matrix.png')
```

**Kya karta hai:**  
Confusion matrix ek grid hai jo dikhata hai:
- Diagonal cells (top-left to bottom-right) = sahi predictions
- Off-diagonal cells = galat predictions (kaunsi category kaunsi se confuse ho rahi hai)

**Saved:** `outputs/charts/classification_confusion_matrix.png`

---

## Cell 14 — Top Words per Category

```python
feature_names = tfidf.get_feature_names_out()

for i, category in enumerate(model.classes_):
    top_indices = np.argsort(model.coef_[i])[-15:]
    top_words = [feature_names[j] for j in top_indices]
    axes[i].barh(top_words, top_scores)
plt.savefig('../outputs/charts/top_words_per_category.png')
```

**Kya karta hai:**  
Har category ke liye top 15 most important words dikhata hai.

**Kaise kaam karta hai:**  
Logistic Regression mein har word ka ek coefficient hota hai per category. Jis word ka coefficient sabse high hai, wo us category ke liye sabse important hai.

**Example:**
- "Bug Fix" category → top words: "error", "fix", "crash", "debug"
- "Feature" category → top words: "implement", "add", "new", "feature"

**Saved:** `outputs/charts/top_words_per_category.png`

---

---

# PART 3 — Topic Modeling (LDA)

**Goal:** Task descriptions se automatically hidden topics nikalna — bina kisi label ke.

---

## Cell 15 — Text Cleaning

```python
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)   # numbers aur punctuation remove
    text = re.sub(r'\s+', ' ', text).strip()
    return text

tasks['clean_desc'] = tasks['Task_Description'].apply(clean_text)
```

**Kya karta hai:**  
Task descriptions clean karta hai:
1. Lowercase convert karta hai
2. Numbers aur special characters remove karta hai
3. Extra spaces remove karta hai

**Kyun clean karna zaroori hai?**  
LDA words par kaam karta hai — "Bug", "bug", "BUG" ko same word treat karna chahiye.

---

## Cell 16 — Count Vectorization

```python
cv = CountVectorizer(max_features=3000, stop_words='english', min_df=5)
doc_term_matrix = cv.fit_transform(tasks['clean_desc'])
print('Doc-term matrix shape:', doc_term_matrix.shape)
```

**Kya karta hai:**  
Text ko word count matrix mein convert karta hai.

**TF-IDF se alag kyun?**  
LDA ko raw word counts chahiye, TF-IDF weights nahi. LDA probability-based model hai.

**Parameters:**
- `max_features=3000` — top 3000 words
- `min_df=5` — sirf wo words jo kam se kam 5 documents mein aaye (rare words ignore)

**Output:** `(10000, 3000)` — 10,000 tasks, 3000 words

---

## Cell 17 — LDA Model Training

```python
NUM_TOPICS = 8
lda = LatentDirichletAllocation(n_components=NUM_TOPICS, random_state=42, max_iter=20)
lda.fit(doc_term_matrix)
```

**Kya karta hai:**  
LDA (Latent Dirichlet Allocation) model train karta hai.

**LDA kya hai?**  
LDA ek unsupervised algorithm hai jo documents mein hidden topics dhundta hai. Ye assume karta hai:
- Har document multiple topics ka mixture hai
- Har topic certain words ka mixture hai

**Kyun 8 topics?**  
Task management mein typically 8 broad areas hote hain (bugs, features, testing, deployment, etc.)

---

## Cell 18 — Topic Words Print

```python
vocab = cv.get_feature_names_out()

for topic_idx, topic in enumerate(lda.components_):
    top_words = [vocab[i] for i in topic.argsort()[-10:][::-1]]
    print(f'Topic {topic_idx + 1}: {" | ".join(top_words)}')
```

**Kya karta hai:**  
Har topic ke top 10 words print karta hai.

**Kaise kaam karta hai:**  
`lda.components_` ek matrix hai jisme har topic ke liye har word ka weight hai. Highest weight wale words wo topic represent karte hain.

**Example output:**
```
Topic 1: bug | fix | error | crash | issue | resolve | debug | test | code | review
Topic 2: deploy | server | release | production | build | pipeline | docker | ci | cd | environment
```

---

## Cell 19 — Dominant Topic Assignment

```python
topic_probs = lda.transform(doc_term_matrix)
tasks['dominant_topic'] = topic_probs.argmax(axis=1) + 1
print(tasks['dominant_topic'].value_counts())
```

**Kya karta hai:**

1. `lda.transform()` — har task ke liye 8 topics ki probability nikalta hai
2. `argmax(axis=1)` — sabse high probability wala topic select karta hai
3. `+1` — topics 0-indexed hain, 1-indexed banana ke liye

**Example:**
```
Task: "Fix login bug causing crash"
Topic probabilities: [0.6, 0.1, 0.05, 0.1, 0.05, 0.05, 0.05, 0.0]
Dominant topic: 1 (Bug Fix)
```

---

## Cell 20 — Topic Distribution Chart

```python
tasks['dominant_topic'].value_counts().sort_index().plot(kind='bar', color='coral')
plt.savefig('../outputs/charts/topic_distribution.png')
```

**Kya karta hai:**  
Bar chart banata hai — kitne tasks har topic mein hain.

**Saved:** `outputs/charts/topic_distribution.png`

---

## Cell 21 — Topic Word Clouds

```python
for topic_idx, topic in enumerate(lda.components_):
    word_freq = {vocab[i]: topic[i] for i in topic.argsort()[-50:]}
    wc = WordCloud(...).generate_from_frequencies(word_freq)
    axes[topic_idx].imshow(wc)
plt.savefig('../outputs/charts/topic_wordclouds.png')
```

**Kya karta hai:**  
Har topic ke liye ek word cloud banata hai. Word ka size = us topic mein uska weight.

**Saved:** `outputs/charts/topic_wordclouds.png`

---

## Cell 22 — Topic vs Category Analysis

```python
topic_category = tasks.groupby(['dominant_topic', 'Category']).size().unstack(fill_value=0)
topic_category.plot(kind='bar', stacked=True)
plt.savefig('../outputs/charts/topic_vs_category.png')
```

**Kya karta hai:**  
Stacked bar chart banata hai jo dikhata hai:
- Har topic mein kaun kaun si categories hain
- Kya LDA ke topics original categories se match karte hain

**Saved:** `outputs/charts/topic_vs_category.png`

---

---

## Summary — Saare Outputs

| Chart File | Kya Dikhata Hai |
|-----------|-----------------|
| `sentiment_comparison.png` | Original vs VADER sentiment distribution |
| `sentiment_wordclouds.png` | Word clouds for Positive/Neutral/Negative comments |
| `classification_confusion_matrix.png` | Model ki prediction accuracy per category |
| `top_words_per_category.png` | Har category ke most important words |
| `topic_distribution.png` | Kitne tasks har topic mein hain |
| `topic_wordclouds.png` | Har LDA topic ke top words visually |
| `topic_vs_category.png` | Topics aur original categories ka overlap |

---

## Flow Diagram

```
DATA LOAD
    │
    ├── comments.csv (30,000)
    │       │
    │       └── PART 1: Sentiment Analysis
    │               ├── VADER se predict karo
    │               ├── Original labels se compare karo
    │               └── Charts save karo
    │
    └── tasks.csv (10,000)
            │
            ├── PART 2: Text Classification
            │       ├── Title + Description combine karo
            │       ├── TF-IDF se vectorize karo
            │       ├── Logistic Regression train karo
            │       └── Confusion matrix + top words
            │
            └── PART 3: Topic Modeling
                    ├── Text clean karo
                    ├── Count Vectorizer se vectorize karo
                    ├── LDA se 8 topics nikalo
                    └── Topic distribution + wordclouds
```
