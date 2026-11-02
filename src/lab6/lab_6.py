import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Save figures to files instead of opening windows
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import adjusted_rand_score, classification_report, confusion_matrix

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 40)
pd.set_option("display.max_columns", 20)

# ------------------------------------------------------------
# Task 1. Load the Individual Dataset
# ------------------------------------------------------------
df = pd.read_csv("clean_dataset.csv")
print("=== Task 1. Dataset loaded ===")
print(f"Shape: {df.shape}")
print(df.drop(columns=["video_url"]).head())  # URL column is long and not needed for viewing
print()
print(df.dtypes)
print()

# ------------------------------------------------------------
# Task 2. Select Features and Validate Data
# ------------------------------------------------------------
# Continuous numeric features of a video. The first two are used for the 2D plots.
features = ["views_count", "day_since_published", "title_length"]
X = df[features].copy()
X = X.apply(pd.to_numeric, errors="raise")  # Fails loudly if a column is not numeric

print("=== Task 2. Selected features ===")
print(f"Features: {features}")
print(f"Missing values per feature: {X.isna().sum().to_dict()}")
if X.isna().any().any():
    raise ValueError("Selected features contain missing values")
if len(X) < 10:
    raise ValueError("At least 10 rows are required")
print("Validation passed: no missing values, enough rows\n")
print(X.describe().round(2))
print()

# ------------------------------------------------------------
# Task 3. Standardize the Features
# ------------------------------------------------------------
# z = (x - mean) / std  ->  views_count (millions) no longer dominates title_length (tens)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("=== Task 3. Standardization ===")
print("Scaled data shape:", X_scaled.shape)
print(f"Means after scaling: {X_scaled.mean(axis=0).round(4)}")
print(f"Std after scaling:   {X_scaled.std(axis=0).round(4)}\n")

# ------------------------------------------------------------
# Task 4. Detect Anomalies with Isolation Forest
# ------------------------------------------------------------
iso = IsolationForest(contamination=0.10, random_state=42)
iso_pred = iso.fit_predict(X_scaled)  # -1 = outlier, +1 = inlier
df["IF_anomaly"] = (iso_pred == -1).astype(int)  # 1 = anomaly, 0 = normal
df["IF_score"] = -iso.score_samples(X_scaled)  # Higher = more anomalous

print("=== Task 4. Isolation Forest ===")
print(df["IF_anomaly"].value_counts().rename("count"))
print()

# ------------------------------------------------------------
# Task 5. Detect Anomalies with LOF
# ------------------------------------------------------------
lof = LocalOutlierFactor(n_neighbors=min(20, len(X) - 1), contamination=0.10)
lof_pred = lof.fit_predict(X_scaled)
df["LOF_anomaly"] = (lof_pred == -1).astype(int)
df["LOF_score"] = -lof.negative_outlier_factor_  # ~1 = normal density, >1 = sparser than neighbours

print("=== Task 5. Local Outlier Factor ===")
print(df["LOF_anomaly"].value_counts().rename("count"))
print()

# ------------------------------------------------------------
# Task 6. Compare the Two Algorithms
# ------------------------------------------------------------
summary = pd.DataFrame({
    "Algorithm": ["Isolation Forest", "Local Outlier Factor"],
    "Anomalies": [df["IF_anomaly"].sum(), df["LOF_anomaly"].sum()],
    "Anomaly rate": [df["IF_anomaly"].mean(), df["LOF_anomaly"].mean()],
})
ari = adjusted_rand_score(df["IF_anomaly"], df["LOF_anomaly"])

both = ((df["IF_anomaly"] == 1) & (df["LOF_anomaly"] == 1)).sum()
only_if = ((df["IF_anomaly"] == 1) & (df["LOF_anomaly"] == 0)).sum()
only_lof = ((df["IF_anomaly"] == 0) & (df["LOF_anomaly"] == 1)).sum()

print("=== Task 6. Comparison ===")
print(summary.to_string(index=False))
print(f"\nAgreement (ARI): {ari:.4f}")
print(f"Flagged by both: {both} | only IF: {only_if} | only LOF: {only_lof}")
print("\nCross-table (rows = IF, columns = LOF):")
print(pd.crosstab(df["IF_anomaly"], df["LOF_anomaly"], rownames=["IF"], colnames=["LOF"]))

show_cols = ["id", "video_title"] + features + ["IF_anomaly", "IF_score", "LOF_anomaly", "LOF_score"]
flagged = df[(df["IF_anomaly"] == 1) | (df["LOF_anomaly"] == 1)][show_cols].copy()
flagged["video_title"] = flagged["video_title"].str.slice(0, 35)
print("\nObservations flagged by at least one algorithm (sorted by IF_score):")
print(flagged.sort_values("IF_score", ascending=False).round(3).to_string(index=False))
print()

# ------------------------------------------------------------
# Task 7. Visualize Detected Anomalies
# ------------------------------------------------------------
print("=== Task 7. Visualization ===")
# Only a 2D projection of the 3D feature space (title_length is not shown)
for column, title, filename in [
    ("IF_anomaly", "Isolation Forest: anomaly flags", "if_anomalies.png"),
    ("LOF_anomaly", "LOF: anomaly flags", "lof_anomalies.png"),
]:
    plt.figure(figsize=(7, 5))
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=df[column], cmap="coolwarm", s=35)
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title(title)
    plt.colorbar(label="Anomaly (1 = yes)")
    plt.tight_layout()
    plt.savefig(filename, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Plot saved as '{filename}'")

# Extra: same projection with a log x-axis, because views_count is very skewed
# and most normal points are squeezed against the left edge on a linear axis
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, column, title in zip(axes, ["IF_anomaly", "LOF_anomaly"], ["Isolation Forest", "LOF"]):
    normal = df[column] == 0
    ax.scatter(X.loc[normal, features[0]], X.loc[normal, features[1]], c="tab:blue", s=30, label="normal (0)")
    ax.scatter(X.loc[~normal, features[0]], X.loc[~normal, features[1]], c="tab:red", s=55, marker="X", label="anomaly (1)")
    ax.set_xscale("log")
    ax.set_xlabel(f"{features[0]} (log scale)")
    ax.set_title(f"{title}: anomaly flags")
    ax.legend()
axes[0].set_ylabel(features[1])
plt.tight_layout()
plt.savefig("anomalies_log_scale.png", dpi=100, bbox_inches="tight")
plt.close()
print("Plot saved as 'anomalies_log_scale.png'\n")

# ------------------------------------------------------------
# Task 8. Check the Meaning of the label Column
# ------------------------------------------------------------
print("=== Task 8. label column check ===")
if "label" in df.columns:
    print("The existing label is NOT automatically an anomaly label.")
    print("Evaluate against label only if its meaning is verified.")
    # In this dataset label was assigned manually in Lab 2 as a binary
    # classification target with a 50/50 split - it does not mark rare/abnormal
    # videos, so Precision/Recall/F1 against it would be meaningless.
    print(f"label distribution: {df['label'].value_counts().to_dict()} (50/50 - not a rare-event label)")
    print("\nFor information only - how flags overlap with label (NOT an evaluation):")
    print(pd.crosstab(df["label"], df["IF_anomaly"], rownames=["label"], colnames=["IF_anomaly"]))
    # If label=1 truly denoted an anomaly, we would uncomment:
    # print(classification_report(df["label"], df["IF_anomaly"], zero_division=0))
    # print(confusion_matrix(df["label"], df["IF_anomaly"]))
print()

# ------------------------------------------------------------
# Task 9. Save Results
# ------------------------------------------------------------
df.to_csv("anomaly_results.csv", index=False)
print("=== Task 9. Results saved ===")
print("Saved: anomaly_results.csv")
print(f"Columns added: {['IF_anomaly', 'IF_score', 'LOF_anomaly', 'LOF_score']}")
