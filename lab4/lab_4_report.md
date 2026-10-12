# Laboratory Work No. 4–5: Binary Classification Models

**Course:** Artificial Intelligence in Cybersecurity  
**Variant:** Individual dataset based on lab2-lab3 work

## 1. Objective
Develop and compare binary classification models (Decision Tree, Logistic Regression, Random Forest, KNN) on the individual dataset created in Lab 2 and processed in Lab 3. Evaluate performance and select the most suitable model.

## 2. Dataset Description
- **Source:** `clean_dataset.csv` (100 rows, produced in Lab 3)
- **Features (X):** `title_length`, `contains_number`, `day_since_published`, `views_count`, `has_uppercase_words`
- **Target (y):** `label` (binary: 0 or 1)
- **Class distribution:** Imbalanced — see `label.value_counts()` below

| Class | Count | Percentage |
|-------|-------|------------|
| True (1)   | 50    | 50.0%      |
| False (0)  | 50    | 50.0%      |

*(Balanced classes in this dataset; the methodological note still warns that small manually-labeled datasets can produce artificially high scores.)*

## 3. Data Preparation
- Features selected: all 5 numeric columns from `clean_dataset.csv`
- Train/test split: 80% train, 20% test with `stratify=y` to preserve class proportions
- `StandardScaler` fitted only on training data, then applied to test data (required for Logistic Regression and KNN)

## 4. Models Trained and Evaluated

### 4.1 Decision Tree
- `max_depth=3`, `random_state=42`
- Visualization saved as `decision_tree.png`
- Metrics: Accuracy=1.0000, Precision=1.0000, Recall=1.0000, F1-score=1.0000

### 4.2 Logistic Regression
- `max_iter=1000`, `random_state=42`
- `StandardScaler` applied
- Coefficients: title_length=-0.1111, contains_number=0.1688, day_since_published=0.1703, views_count=2.8233, has_uppercase_words=0.2081
- Intercept: 0.5932
- Metrics: Accuracy=0.9000, Precision=0.9000, Recall=0.9000, F1-score=0.9000

### 4.3 Random Forest
- `n_estimators=100`, `random_state=42`
- Feature importances: views_count=0.8166, day_since_published=0.0983, title_length=0.0581, has_uppercase_words=0.0189, contains_number=0.0079
- Metrics: Accuracy=1.0000, Precision=1.0000, Recall=1.0000, F1-score=1.0000

### 4.4 K-Nearest Neighbors (KNN)
- `n_neighbors=5`
- Requires feature scaling (using the same scaler from training data)
- Metrics: Accuracy=0.8000, Precision=0.7500, Recall=0.9000, F1-score=0.8182

## 5. Comparison Table

| Model                | Accuracy | Precision | Recall | F1-score |
|----------------------|----------|-----------|--------|----------|
| Decision Tree        | 1.0000   | 1.0000    | 1.0000 | 1.0000   |
| Random Forest        | 1.0000   | 1.0000    | 1.0000 | 1.0000   |
| Logistic Regression  | 0.9000   | 0.9000    | 0.9000 | 0.9000   |
| KNN                  | 0.8000   | 0.7500    | 0.9000 | 0.8182   |

*(Sorted by F1-score descending)*

**Figure: F1-score comparison** — bar chart saved as `f1_comparison.png`

**Best model by F1-score:** Decision Tree (tied with Random Forest at F1=1.0; Decision Tree listed first)

**Confusion matrix** for the best model (Decision Tree) — saved as `confusion_matrix.png`

*(Visuals are PNG files generated alongside the script execution.)*

## 6. Key Findings
- The model with the highest F1-score is **Decision Tree** (F1=1.0000), tied with **Random Forest** (F1=1.0000). Decision Tree is listed first due to sort ordering.
- **Accuracy ranking** differs from F1-score ranking: Decision Tree and Random Forest both have 1.0 accuracy, while Logistic Regression has 0.9 and KNN has 0.8.
- **Most important feature** (from Random Forest feature importances): `views_count` (0.8166), followed by `day_since_published` (0.0983), `title_length` (0.0581), `has_uppercase_words` (0.0189), and `contains_number` (0.0079).
- **Limitations**: Small dataset (~100 rows with balanced classes), manually assigned labels — results are an educational experiment per lab guidelines, not production-ready. The perfect scores (1.0) for Decision Tree and Random Forest should be interpreted cautiously given the small, manually-labeled dataset.

## 7. Summary and Report Guidance (skipping cover page & screenshots)
- Key findings are summarized above: best model, feature importances, and limitations.
- The comparison table and metrics are documented; figures (`decision_tree.png`, `f1_comparison.png`, `confusion_matrix.png`) were generated during execution.
- Discuss which errors (FP or FN) would be more serious for your cybersecurity application context, per the lab guidelines.
- Reference the methodological note: small dataset (~100 rows) and manually assigned labels can produce unstable or artificially high scores; treat results as an educational experiment.

---

**Generated:** `lab4.py` script with all model training, evaluation, and visualization code.  
**Output files:** `decision_tree.png`, `f1_comparison.png`, `confusion_matrix.png`, `clean_dataset.csv` (already existed).