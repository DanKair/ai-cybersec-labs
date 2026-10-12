import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# ------------------------------------------------------------
# 1. Load the Dataset
# ------------------------------------------------------------
df = pd.read_csv("clean_dataset.csv")
print("=== Dataset loaded ===")
print(df.head())
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")

# ------------------------------------------------------------
# 2. Select Features and Target
# ------------------------------------------------------------
# Numeric features from the dataset
features = ["title_length", "contains_number", "day_since_published", "views_count", "has_uppercase_words"]
X = df[features].copy()
y = df["label"].copy()

print(f"Features: {X.shape}")
print(f"Target: {y.shape}")
print(f"Classes distribution: {y.value_counts().to_dict()}\n")

# ------------------------------------------------------------
# 3. Split into Training and Test Sets
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train: {X_train.shape} | Test: {X_test.shape}\n")

# ------------------------------------------------------------
# 4. Scale features (fit on train only) — important for LR and KNN
# ------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------
# 5. Build a Decision Tree
# ------------------------------------------------------------
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("=== Decision Tree ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_dt, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_dt, zero_division=0):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred_dt, zero_division=0):.4f}\n")

# Visualize the tree
plt.figure(figsize=(13, 7))
plot_tree(dt, feature_names=features, class_names=["0", "1"], filled=True, rounded=True, fontsize=9)
plt.title("Decision Tree")
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=100, bbox_inches="tight")
plt.close()
print("Decision tree visualization saved as 'decision_tree.png'\n")

# ------------------------------------------------------------
# 6. Train Logistic Regression
# ------------------------------------------------------------
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

print("=== Logistic Regression ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lr, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_lr, zero_division=0):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred_lr, zero_division=0):.4f}")
print(f"Coefficients: {dict(zip(features, lr.coef_[0]))}")
print(f"Intercept:   {lr.intercept_[0]:.4f}\n")

# ------------------------------------------------------------
# 7. Train a Random Forest
# ------------------------------------------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("=== Random Forest ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_rf, zero_division=0):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred_rf, zero_division=0):.4f}")
print("Feature importances:")
for name, value in zip(features, rf.feature_importances_):
    print(f"  {name}: {value:.4f}\n")

# ------------------------------------------------------------
# 8. Apply K-Nearest Neighbors (KNN)
# ------------------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

print("=== KNN ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_knn):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_knn, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_knn, zero_division=0):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred_knn, zero_division=0):.4f}\n")

# ------------------------------------------------------------
# 9. Evaluate Models and Build a Comparison Table
# ------------------------------------------------------------
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
print("=== Comparison Table (sorted by F1-score) ===")
print(results_df.round(4).to_string(index=False) + "\n")

# ------------------------------------------------------------
# 10. Visualize F1-score comparison
# ------------------------------------------------------------
plt.figure(figsize=(9, 5))
results_df.plot.bar(x="Model", y="F1-score", legend=False, rot=20)
plt.ylim(0, 1)
plt.title("F1-score comparison")
plt.ylabel("F1-score")
plt.tight_layout()
plt.savefig("f1_comparison.png", dpi=100, bbox_inches="tight")
plt.close()
print("F1-score comparison chart saved as 'f1_comparison.png'\n")

# ------------------------------------------------------------
# 11. Confusion Matrix for the best model
# ------------------------------------------------------------
best_name = results_df.iloc[0]["Model"]
print(f"Best model by F1-score: {best_name}")

plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, predictions[best_name], labels=[0, 1], display_labels=["0", "1"], cmap="Blues"
)
plt.title(f"Confusion matrix: {best_name}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=100, bbox_inches="tight")
plt.close()
print("Confusion matrix saved as 'confusion_matrix.png'\n")

# ------------------------------------------------------------
# 12. Feature importance from Random Forest
# ------------------------------------------------------------
print("=== Random Forest Feature Importances ===")
for name, value in zip(features, rf.feature_importances_):
    print(f"  {name}: {value:.4f}")

# ------------------------------------------------------------
# 13. Summary / Conclusion
# ------------------------------------------------------------
print("\n=== Summary ===")
print(f"Best model: {best_name} with F1-score={results_df.iloc[0]['F1-score']:.4f}")
print("\nNote: Small dataset (~100 rows) may produce unstable results;")
print("treat as educational experiment per lab guidelines.")