# Lab 4-5: Machine Learning Models Explanation

*Explained for someone who knows pandas & numpy but is new to machine learning.*

## Overview

This lab implements **binary classification** using 4 different models on your dataset from Labs 2-3. The goal is to predict a binary `label` (0/1) based on 5 video features.

**Dataset:** `clean_dataset.csv` (100 rows, balanced 50/50 classes)
**Features (X):** 5 columns used as inputs
**Target (y):** `label` column - what we're predicting

---

## Step 1: Load Data 

```python
df = pd.read_csv("clean_dataset.csv")
```

- Reads the CSV into a DataFrame (just like lab2/lab3)
- Output: `Shape: (100, 9)` means 100 rows, 9 columns

**Columns in dataset:**
```
id, video_url, video_title, title_length, contains_number,
day_since_published, views_count, has_uppercase_words, label
```

---

## Step 2: Select Features and Target

```python
features = ["title_length", "contains_number", "day_since_published", "views_count", "has_uppercase_words"]
X = df[features].copy()    # INPUT features (like selecting columns in pandas)
y = df["label"].copy()     # TARGET we want to predict
```

**Simple analogy:**

| Concept | Pandas analogy | Meaning |
|---------|---------------|---------|
| `X` | `df[["col1", "col2"]]` | The columns we'll use to predict |
| `y` | `df["target_column"]` | What we're trying to predict |

**The 5 features selected:**
1. `title_length` - Title length (number of characters)
2. `contains_number` - Does title have a number? (True/False)
3. `day_since_published` - Days since video was posted (number)
4. `views_count` - How many views the video got (number)
5. `has_uppercase_words` - Does title have uppercase words? (True/False)

**Target `y`:** `label` - Binary classification (0 or 1, True or False)

**Output:** `Classes distribution: {True: 50, False: 50}` → Balanced dataset

---

## Step 3: Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
```

**Simple explanation:**

- We need to teach the model on some data, then check if it works on new, unseen data
- `test_size=0.20`: 20% of data goes to testing, 80% to training
- `random_state=42`: Makes the split reproducible (same every time you run)
- `stratify=y`: Keeps the 50/50 class ratio in both training and test sets

**Result:** `Train: (80, 5)` and `Test: (20, 5)` → 80 training samples, 20 test samples

**Why not use all data for training?** If we do, we can't know if the model actually works on new data!

---

## Step 4: Feature Scaling

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Simple explanation (the trickiest part):**

Imagine you have two features:
- "Title length": ranges from ~20 to ~80 characters
- "Views count": ranges from ~1 to ~7,000,000

The numbers have **very different ranges**! If we use them directly, the model might pay attention only to the feature with larger numbers (views count) and ignore the other one.

**Scaling makes them comparable** - like converting everything to a standardized scale (z-scores or 0-1 range).

**Crucial rule:** We fit the scaler on training data only, then use that same scaler on test data. We can't "peek" at test data while training!

**Which models need it?**
- ✅ **Logistic Regression** and **KNN** are sensitive to feature scales
- ✅ **Decision Tree** and **Random Forest** are NOT affected by scaling

**Output:** No direct output, but `X_train_scaled` and `X_test_scaled` are created for the models that need them.

---

## Step 5: Decision Tree

```python
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
```

**Simple analogy:**

Think of a decision tree as a series of **yes/no questions** to classify something:

1. "Does the title contain a number?" → If yes, go to question 2...
2. "Is the title longer than 50 characters?" → If yes, go to question 3...
3. Eventually you reach a "leaf" that says: "This video is class 0" or "This video is class 1"

**`max_depth=3`** limits the tree to only 3 levels of questions (prevents it from becoming overly complex).

**`random_state=42`** makes it reproducible (same tree every time).

**The model then predicts** on the test set: `y_pred_dt = dt.predict(X_test)`

**Metrics printed:**
- **Accuracy:** Overall % of correct predictions
- **Precision:** Of videos predicted as "class 1", how many really are class 1?
- **Recall:** Of all actual class 1 videos, how many did we find?
- **F1-score:** Harmonic mean of precision and recall (balance between the two)

**Visualization:** Also saves a tree picture as `decision_tree.png`

**Result:** Perfect scores (1.0) on all metrics for this dataset

---

## Step 6: Logistic Regression

```python
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
```

**Simple analogy:**

Think of this as **drawing a decision boundary** (a line in 2D, or a plane in higher dimensions) that separates the two classes.

The model calculates: `score = weight1*x1 + weight2*x2 + ... + bias`

- If score > 0 → predict class 1
- If score ≤ 0 → predict class 0

**What's new here:**
- **Must use scaled features** (`X_train_scaled`, `X_test_scaled`)
- The code also prints **coefficients** (weights) and **intercept** (baseline)

**Coefficients you saw:**
```
title_length: -0.1111
contains_number: 0.1688
day_since_published: 0.1703
views_count: 2.8233    ← Biggest weight!
has_uppercase_words: 0.2081
```

**Interpretation (simple):**
- `views_count` has the strongest impact (weight 2.82) - more views makes class 1 more likely
- `title_length` has negative weight (-0.11) - longer titles slightly make class 1 less likely
- The intercept (0.5932) is like the baseline probability

**Result:** Accuracy 0.9, F1-score 0.9

---

## Step 7: Random Forest

```python
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
```

**Simple analogy:**

Instead of **one decision tree**, we grow **100 different trees** (`n_estimators=100`).

Each tree votes on what class the data should be. The final prediction is the **majority vote** - like asking 100 people instead of just 1.

Each tree is slightly different (due to randomness), making the model more robust than a single tree.

**Feature importances you saw:**
```
views_count: 0.8166    ← Most important (81.66%)
day_since_published: 0.0983
title_length: 0.0581
has_uppercase_words: 0.0189
contains_number: 0.0079
```

**Interpretation:** The forest says `views_count` is by far the most important feature.

**Result:** Perfect scores (1.0) on all metrics

---

## Step 8: K-Nearest Neighbors (KNN)

```python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
```

**Simple analogy:**

Imagine you see a **new video** and want to classify it:

1. KNN looks at the **5 most similar videos** from the training set (`n_neighbors=5`)
2. Similarity is based on **Euclidean distance** (like measuring distance in space)
3. It takes the majority class among those 5 neighbors and assigns it

**Critical requirement:** **Must use scaled features!**
- Because KNN measures "distance," features with larger ranges dominate
- If one feature ranges 0-1000 and another 0-1, the 0-1000 feature would always "win"
- Scaling fixes this so all features contribute equally

**Result you saw:** Lower performance (F1=0.8182) compared to other models

---

## Step 9: Comparison Table

```python
predictions = {
    "Decision Tree": y_pred_dt,
    "Logistic Regression": y_pred_lr,
    "Random Forest": y_pred_rf,
    "KNN": y_pred_knn,
}

rows = []
for name, y_pred in predictions.items():
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
    })

results_df = pd.DataFrame(rows).sort_values("F1-score", ascending=False)
```

**Result you saw:**

| Model                | Accuracy | Precision | Recall | F1-score |
|----------------------|----------|-----------|--------|----------|
| Decision Tree        | 1.0000   | 1.0000    | 1.0000 | 1.0000   |
| Random Forest        | 1.0000   | 1.0000    | 1.0000 | 1.0000   |
| Logistic Regression  | 0.9000   | 0.9000    | 0.9000 | 0.9000   |
| KNN                  | 0.8000   | 0.7500    | 0.9000 | 0.8182   |

**Sorted by F1-score (highest first)**

**`zero_division=0`**: Avoids errors when no positive predictions are made (handles edge cases gracefully)

---

## Step 10: F1-score Visualization

```python
results_df.plot.bar(x="Model", y="F1-score", legend=False, rot=20)
plt.savefig("f1_comparison.png")
```

- Creates a **bar chart** comparing F1-scores across all 4 models
- Saved as `f1_comparison.png`
- Easy to see at a glance which model performed best

---

## Step 11: Confusion Matrix

```python
best_name = results_df.iloc[0]["Model"]  # "Decision Tree"

ConfusionMatrixDisplay.from_predictions(
    y_test, predictions[best_name], labels=[0, 1], display_labels=["0", "1"], cmap="Blues"
)
plt.savefig("confusion_matrix.png")
```

**Simple explanation:**

A **2x2 grid** showing:

| | **Predicted: 0** | **Predicted: 1** |
|---|---|---|
| **Actual: 0** | True Negatives (TN) | False Positives (FP) |
| **Actual: 1** | False Negatives (FN) | True Positives (TP) |

**Terms:**
- **True Positives (TP):** Correctly predicted as class 1
- **True Negatives (TN):** Correctly predicted as class 0
- **False Positives (FP):** Wrongly predicted as class 1 (Type I error/"false alarm")
- **False Negatives (FN):** Wrongly predicted as class 0 (Type II error/"missed detection")

- For the Decision Tree (best model), this was saved as `confusion_matrix.png`

---

## Step 12: Feature Importances (from Random Forest)

```python
for name, value in zip(features, rf.feature_importances_):
    print(f"  {name}: {value:.4f}")
```

- Ranks features by how much they contribute to predictions
- Based on how much each feature decreases "impurity" across all 100 trees
- Output: `views_count: 0.8166` (most important) through `contains_number: 0.0079` (least important)

---

## Step 13-14: Summary

```python
print(f"Best model: {best_name} with F1-score={results_df.iloc[0]['F1-score']:.4f}")
print("\nNote: Small dataset (~100 rows) may produce unstable results;")
print("treat as educational experiment per lab guidelines.")
```

- Tells you which model was best (Decision Tree, F1=1.0)
- Includes the **methodological note** from the lab spec about small datasets
- Reminds you to treat results as **educational, not production-ready**

---

## Quick: Which Model Uses What?

| Model | Uses scaled features? | Key characteristic |
|-------|----------------------|-------------------|
| Decision Tree | ❌ No | Tree structure, splits on thresholds |
| Logistic Regression | ✅ Yes | Linear decision boundary, coefficients interpretable |
| Random Forest | ❌ No (internally handles it) | Ensemble of 100 trees, feature importances |
| KNN | ✅ Yes | Distance-based, k nearest neighbors |

---

## Key Takeaways

1. **Data split:** 80% train, 20% test with preserved class ratios
2. **Scaling:** Required for LR and KNN, not for tree models
3. **Model performance:** Decision Tree & Random Forest perfect (1.0), but note the dataset size warning
4. **Most important feature:** `views_count` (according to Random Forest)
5. **Evaluation:** Look at F1-score (not just accuracy), especially with imbalanced classes
6. **Small dataset warning:** ~100 rows with manual labels can give artificially high scores

---

*Generated from `lab4.py` following the Lab Work 4-5 procedure for binary classification.*