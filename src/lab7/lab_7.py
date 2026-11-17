"""Lab 7: Intelligent detection and risk assessment of viral ("abnormal-popularity") videos.

Individual variant: self-improvement YouTube videos (dataset from Labs 2-3).
"Threat" / positive class = a viral video (label = True, i.e. > 100K views).

Why a viral video is treated as a security event: a sudden spike in popularity is the
moment a video (and its channel) attracts view-botting / fake engagement, scam and
spam comment bots, impersonating channels and account-takeover attempts. The system
below plays the role of an IDS for a content platform: it flags such videos early and
ranks them for investigation.
"""
# ---------------- Libraries used in this lab ----------------
# pandas     -> works with tables ("DataFrames"), like Excel inside Python
# numpy      -> fast math on arrays of numbers (log, rounding, conditions)
# matplotlib -> draws charts and saves them as PNG images
# sklearn    -> scikit-learn, the main machine-learning library: models, scaling, metrics
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # "Agg" = draw charts straight into files, never open a window
import matplotlib.pyplot as plt

# train_test_split  -> cuts the table into a training part and a test part
# StratifiedKFold   -> recipe for cross-validation (several train/test splits in a row)
# cross_val_score   -> trains + tests a model on every split of that recipe
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
# StandardScaler -> rescales every column to mean 0 and standard deviation 1
from sklearn.preprocessing import StandardScaler
# RandomForestClassifier -> many decision trees that vote together
from sklearn.ensemble import RandomForestClassifier
# LogisticRegression -> a simple linear model that outputs a probability (0..1)
from sklearn.linear_model import LogisticRegression
# make_pipeline -> glues several steps (e.g. scaler + model) into one object
from sklearn.pipeline import make_pipeline
# Metrics = numbers that tell how good the predictions are (explained in Task 4)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)

# Make pandas print wide tables on one line instead of cutting them
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 38)  # cut long video titles to 38 characters

# Many ML steps use randomness (shuffling rows, building random trees).
# Fixing the "seed" to 42 makes every run give exactly the same result,
# so the numbers in the report can be reproduced.
RANDOM_STATE = 42

# ------------------------------------------------------------
# Common data preparation (shared by all tasks)
# ------------------------------------------------------------
# Read the CSV file into a DataFrame: 100 rows (videos) x 9 columns
df = pd.read_csv("clean_dataset.csv")

# ML models only understand numbers. True/False columns are turned into 1/0.
for col in ["contains_number", "has_uppercase_words", "label"]:
    df[col] = df[col].astype(int)  # True -> 1, False -> 0

# Derived feature (a new column computed from existing ones):
# average views per day = how FAST a video gains popularity ("velocity").
# .clip(lower=1) replaces 0 days by 1 so we never divide by zero.
df["views_per_day"] = (df["views_count"] / df["day_since_published"].clip(lower=1)).round(1)

# Features (X) = the inputs the model is allowed to look at.
# Target  (y) = the answer the model must learn to predict (here: viral or not).
#
# IMPORTANT: label was defined in Lab 2 as views_count > 100 000.
# If views_count (or views_per_day, which together with day_since_published
# reconstructs views_count) is given to the model, it simply re-learns that
# threshold and scores 100% - that is data leakage, not detection.
# ("Data leakage" = the answer is secretly hidden inside the inputs.)
# So the ML detector uses only signals known *before* views accumulate:
ML_FEATURES = ["title_length", "contains_number", "has_uppercase_words", "day_since_published"]
TARGET = "label"

# Split the videos into two groups:
#   train_df (80%, 80 videos) -> the model LEARNS from these
#   test_df  (20%, 20 videos) -> the model has NEVER seen these; we use them as
#                                "new events" to check how well it really works
# stratify=... keeps the same 50/50 viral/normal ratio in both groups.
# The same split is reused in Task 1 and Task 4.
train_df, test_df = train_test_split(
    df, test_size=0.20, random_state=RANDOM_STATE, stratify=df[TARGET]
)

# ============================================================
# Task 1. Intelligent detection system (Random Forest + alerts)
# ============================================================
print("=== Task 1. Viral-video detection system (Random Forest) ===")
print(f"Dataset: {df.shape[0]} videos | features X: {ML_FEATURES} | target: {TARGET} (1 = viral)")
print(f"Train: {len(train_df)} videos | Test events: {len(test_df)} videos")

# A Random Forest is a "committee" of decision trees.
# Each tree is a chain of yes/no questions such as "title_length > 50?" and is
# trained on a random sample of the rows; the forest's answer is the majority vote.
#   n_estimators=200   -> build 200 trees
#   max_depth=4        -> each tree asks at most 4 questions in a row (keeps trees simple)
#   min_samples_leaf=2 -> every final answer must be based on at least 2 videos
# The last two settings stop the trees from memorising our small dataset (overfitting).
rf = RandomForestClassifier(
    n_estimators=200, max_depth=4, min_samples_leaf=2, random_state=RANDOM_STATE
)
# .fit(X, y) = TRAINING: the model looks at the inputs and the correct answers
rf.fit(train_df[ML_FEATURES], train_df[TARGET])

# .predict(X)       -> final class for every test video: 0 (normal) or 1 (viral)
# .predict_proba(X) -> probability of each class, shape (20, 2): [P(normal), P(viral)]
#                      [:, 1] takes the second column = P(viral) = share of trees voting "viral"
predictions = rf.predict(test_df[ML_FEATURES])
probabilities = rf.predict_proba(test_df[ML_FEATURES])[:, 1]

# The detector only sees weak pre-view signals, so its probabilities are compressed
# into roughly 0.25-0.70 (it is never 85-100% sure). A fixed 0.85 cut-off from the
# example would never fire, so CRITICAL is set to 0.60 = clearly above the 0.5 boundary.
CRITICAL_THRESHOLD = 0.60  # probability of "viral" above which an alert is CRITICAL


def alert_level(prediction, probability):
    """Turn a model answer into a human-friendly alert, like a real IDS does."""
    if prediction == 0:
        return "INFO"  # model thinks the video is ordinary
    if probability >= CRITICAL_THRESHOLD:
        return "CRITICAL"  # model is confident the video is viral
    return "WARNING"  # viral, but the model is not confident


# Build the event log: one row per test video.
# .values takes the raw numbers out of a pandas column (drops the old row index).
event_log = pd.DataFrame({
    "event_id": range(1, len(test_df) + 1),           # 1, 2, ..., 20
    "video_id": test_df["id"].values,
    "video_title": test_df["video_title"].values,
    "title_length": test_df["title_length"].values,
    "contains_number": test_df["contains_number"].values,
    "has_uppercase_words": test_df["has_uppercase_words"].values,
    "day_since_published": test_df["day_since_published"].values,
    "views_count": test_df["views_count"].values,      # kept for analysis only, NOT a model input
    "views_per_day": test_df["views_per_day"].values,  # kept for analysis only, NOT a model input
    "true_label": test_df[TARGET].values,              # the real answer, to check the model
    "prediction": predictions.astype(int),
    "risk_score": np.round(probabilities, 3),          # P(viral), rounded to 3 decimals
})
# zip(...) walks through two columns side by side; alert_level() is called for each pair
event_log["alert_level"] = [alert_level(p, s) for p, s in zip(event_log["prediction"], event_log["risk_score"])]

show = ["event_id", "video_id", "video_title", "day_since_published", "true_label", "prediction", "risk_score", "alert_level"]
print(event_log[show].to_string(index=False))  # index=False hides pandas' row numbers
# value_counts() counts how many times each alert level appears
print("\nAlerts:", event_log["alert_level"].value_counts().to_dict())
# Feature importance = how much each input helped the trees make decisions (sums to 1)
print("Feature importances:", dict(zip(ML_FEATURES, rf.feature_importances_.round(3))))

# Boolean filtering: keep only rows where the condition is True
flagged = event_log[event_log["prediction"] == 1]
print(f"\nEvents flagged as viral (WARNING/CRITICAL): {len(flagged)} -> video ids {flagged['video_id'].tolist()}")
# true_label is 0/1, so its sum = number of flagged videos that really are viral
print(f"Of them truly viral: {int(flagged['true_label'].sum())} / {len(flagged)}")

event_log.to_csv("event_log.csv", index=False)  # save the table as a CSV file
print("Saved: event_log.csv\n")

# ============================================================
# Task 2. Risk level assessment and response priorities
# ============================================================
print("=== Task 2. Risk assessment ===")
# The lab asks to build on Task 1's results, so we read its output file back
risk = pd.read_csv("event_log.csv")

# Thresholds are plain constants so they are easy to find, justify and change
HIGH_THRESHOLD = 0.60     # same as CRITICAL alert: clearly predicted viral
MEDIUM_THRESHOLD = 0.45   # borderline: model is unsure (close to 0.5) -> worth a look
FRESH_DAYS = 30           # recent uploads are where a spike is still developing


def assess_risk(row):
    """Receives ONE row of the table and returns its risk level.

    The checks go from most to least serious; the first one that matches wins.
    """
    p = row["risk_score"]
    if row["prediction"] == 1 and p >= HIGH_THRESHOLD:
        return "HIGH"
    if row["prediction"] == 1 or p >= MEDIUM_THRESHOLD:
        return "MEDIUM"
    # A fresh video that is not yet predicted viral still gets attention if p is non-trivial
    if row["day_since_published"] <= FRESH_DAYS and p >= 0.30:
        return "MEDIUM"
    return "LOW"


# .apply(func, axis=1) calls assess_risk() once for every ROW (axis=1 = row by row)
risk["risk_level"] = risk.apply(assess_risk, axis=1)

# A dictionary (key -> value) that maps each risk level to the recommended action
RESPONSE = {
    "HIGH": "Verify view authenticity (bot traffic), enable comment spam/scam filter, alert channel owner, enforce 2FA",
    "MEDIUM": "Put on watch-list, re-score in 24h, sample comments for scam links",
    "LOW": "No action, routine monitoring",
}
# .map(dict) replaces every value in the column by the matching dictionary value
risk["response"] = risk["risk_level"].map(RESPONSE)

# Count events per level. reindex(...) forces the order LOW, MEDIUM, HIGH and
# puts 0 for a level that does not appear at all (fill_value=0).
level_order = ["LOW", "MEDIUM", "HIGH"]
counts = risk["risk_level"].value_counts().reindex(level_order, fill_value=0)
print("Criteria: HIGH = predicted viral & p >= 0.60 | MEDIUM = predicted viral, or p >= 0.45,"
      " or fresh (<= 30 days) with p >= 0.30 | LOW = otherwise")
print("Events per risk level:", counts.to_dict())

# Priority order: risk level first, then probability, then fresher video first.
# Text levels can't be sorted by importance ("HIGH" < "LOW" alphabetically!), so we
# add a temporary helper column with numbers: HIGH=2, MEDIUM=1, LOW=0.
risk["_level_rank"] = risk["risk_level"].map({"HIGH": 2, "MEDIUM": 1, "LOW": 0})
# Sort by 3 columns: rank (big first), risk_score (big first), days (small = fresher first)
top5 = risk.sort_values(
    ["_level_rank", "risk_score", "day_since_published"], ascending=[False, False, True]
).head(5)  # .head(5) keeps the first 5 rows
print("\nTOP-5 events by investigation priority:")
print(top5[["event_id", "video_id", "video_title", "day_since_published", "true_label", "risk_score", "risk_level"]].to_string(index=False))
risk = risk.drop(columns="_level_rank")  # remove the helper column, it is not needed in the CSV

print("\nRecommended response per level:")
for level in reversed(level_order):  # print HIGH first
    print(f"  {level:<6} -> {RESPONSE[level]}")  # :<6 pads the text to 6 characters

# Bar chart: one bar per risk level, coloured green / orange / red
colors = {"LOW": "#4caf50", "MEDIUM": "#ff9800", "HIGH": "#e53935"}
plt.figure(figsize=(6, 4))  # canvas size in inches
plt.bar(counts.index, counts.values, color=[colors[l] for l in counts.index])
for x, v in zip(counts.index, counts.values):
    plt.text(x, v + 0.1, str(v), ha="center")  # write the number above each bar
plt.title("Distribution of risk levels (20 test videos)")
plt.xlabel("Risk level")
plt.ylabel("Number of events")
plt.tight_layout()  # avoid labels being cut off
plt.savefig("risk_distribution.png", dpi=100)
plt.close()  # free memory; the picture is already saved
print("Plot saved as 'risk_distribution.png'")

risk.to_csv("risk_analysis.csv", index=False)
print("Saved: risk_analysis.csv\n")

# ============================================================
# Task 3. Rule-based detection from characteristic features
# ============================================================
print("=== Task 3. Feature-based (rule) detection ===")
# Here there is NO machine learning: we write the detection rule ourselves,
# like the malware example (sum of indicators -> compare with a threshold).
# This task works on ALL 100 videos (no train/test split is needed for a fixed rule).
#
# Three indicators of an abnormal popularity spike:
#   views_count   - total volume of activity        (like traffic_volume in an IDS)
#   views_per_day - speed of activity                (like request_count per time unit)
#   clickbait     - title uses numbers or CAPS words (like a phishing "lure")
# views are heavily skewed (182 ... 11.4M), so log1p is applied before min-max
# scaling; otherwise one mega-viral video would squash everything else to ~0.
threat = df[["id", "video_title", "views_count", "views_per_day",
             "contains_number", "has_uppercase_words", "label"]].copy()  # .copy() = independent table


def log_minmax(series):
    """Put a column on a 0..1 scale so different features can be added together.

    Step 1: np.log1p(x) = log(1 + x). Logarithm turns 100 / 10 000 / 1 000 000 into
            roughly 4.6 / 9.2 / 13.8, i.e. it shrinks huge values much more than small ones.
    Step 2: min-max scaling (x - min) / (max - min): smallest value -> 0, largest -> 1.
    """
    s = np.log1p(series)
    return (s - s.min()) / (s.max() - s.min())


threat["volume_score"] = log_minmax(threat["views_count"]).round(3)
threat["velocity_score"] = log_minmax(threat["views_per_day"]).round(3)
# (a + b) > 0 is True when at least one of the two flags is 1 -> "number OR CAPS"
threat["clickbait_score"] = ((threat["contains_number"] + threat["has_uppercase_words"]) > 0).astype(int)

# Weighted sum: each indicator (0..1) is multiplied by its importance; weights add up
# to 1.0, so risk_score is also between 0 and 1.
# Weights: the two activity indicators matter most, title style is a weak hint
W_VOLUME, W_VELOCITY, W_CLICKBAIT = 0.45, 0.45, 0.10
threat["risk_score"] = (
    W_VOLUME * threat["volume_score"]
    + W_VELOCITY * threat["velocity_score"]
    + W_CLICKBAIT * threat["clickbait_score"]
).round(3)

# Threshold choice: try several cut-offs and see what each one would flag.
#   precision_vs_label = of the flagged videos, how many are really viral
#   recall_vs_label    = of all viral videos, how many got flagged
# label is used ONLY here, to justify the threshold - the rule never reads it.
print("Threshold sweep (label shown for reference only, rules do not use it):")
sweep = []
for t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    flag = (threat["risk_score"] >= t).astype(int)  # 1 = would be flagged at this threshold
    sweep.append({"threshold": t, "flagged": int(flag.sum()),
                  "precision_vs_label": round(precision_score(threat["label"], flag, zero_division=0), 3),
                  "recall_vs_label": round(recall_score(threat["label"], flag, zero_division=0), 3)})
print(pd.DataFrame(sweep).to_string(index=False))  # list of dicts -> table

RULE_THRESHOLD = 0.60
# np.where(condition, value_if_true, value_if_false) works on the whole column at once
threat["result"] = np.where(threat["risk_score"] >= RULE_THRESHOLD, "SUSPICIOUS", "NORMAL")
print(f"\nChosen threshold: risk_score >= {RULE_THRESHOLD} -> SUSPICIOUS")
print(threat["result"].value_counts().to_dict())

print("\nTop-10 objects by risk_score:")
print(threat.sort_values("risk_score", ascending=False)[
    ["id", "video_title", "views_count", "views_per_day", "volume_score", "velocity_score",
     "clickbait_score", "risk_score", "result"]
].head(10).to_string(index=False))

# Share of videos where the rule's verdict (SUSPICIOUS=1 / NORMAL=0) equals label.
# Comparing two columns gives True/False per row; .mean() of True/False = share of True.
agree = (threat["result"].eq("SUSPICIOUS").astype(int) == threat["label"]).mean()
print(f"\nAgreement of rules with label (reference only): {agree:.2%}")

threat.to_csv("threat_detection.csv", index=False)
print("Saved: threat_detection.csv\n")

# ============================================================
# Task 4. ML classification (Logistic Regression) + evaluation
# ============================================================
print("=== Task 4. ML classification (Logistic Regression) ===")
# TF-IDF is a way to turn TEXT into numbers. Our features are already numbers,
# so instead we only need to bring them to a common scale.
print("Data is numeric (no free text to classify), so TF-IDF is not used;"
      " features are standardized with StandardScaler instead.")

# Common ML naming: X = inputs (table), y = answers (one column)
X_train, y_train = train_df[ML_FEATURES], train_df[TARGET]
X_test, y_test = test_df[ML_FEATURES], test_df[TARGET]

# Why scale? title_length is ~10..80 while day_since_published is ~1..1945.
# Logistic Regression would treat bigger numbers as more important, so
# StandardScaler converts each column to z = (x - mean) / std (mean 0, std 1).
#
# Logistic Regression: computes a weighted sum of the scaled features,
#   z = w1*x1 + w2*x2 + ... + b, and squeezes it into 0..1 with the sigmoid
#   1 / (1 + e^-z). That is P(viral). If P >= 0.5 -> predict 1 (viral).
#
# Pipeline: scaler is fitted on the training part only, then applied to test.
# (If the scaler saw the test data, information about the "exam" would leak in.)
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
lr.fit(X_train, y_train)           # learns scaler mean/std + the weights w
y_pred = lr.predict(X_test)        # 0 / 1 for each of the 20 unseen videos
y_prob = lr.predict_proba(X_test)[:, 1]  # P(viral) for each of them

# Evaluation metrics (TP/TN/FP/FN explained below at the confusion matrix):
#   Accuracy  = share of all predictions that are correct
#   Precision = TP / (TP + FP): when the model says "viral", how often is it right?
#   Recall    = TP / (TP + FN): of all truly viral videos, how many did it catch?
#   F1-score  = harmonic mean of Precision and Recall (high only if BOTH are high)
# zero_division=0 -> return 0 instead of crashing if a division by zero happens
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1-score": f1_score(y_test, y_pred, zero_division=0),
}
for name, value in metrics.items():
    print(f"{name:<10} {value:.4f}")  # .4f = 4 digits after the decimal point
print("\nClassification report:")
# Same metrics, but computed separately for each class ("support" = number of videos)
print(classification_report(y_test, y_pred, target_names=["normal (0)", "viral (1)"], zero_division=0))

# Coefficients = learned weights w. Positive -> pushes towards "viral",
# negative -> pushes towards "normal". Bigger absolute value = stronger effect.
# named_steps lets us reach the LogisticRegression step inside the pipeline.
coefs = lr.named_steps["logisticregression"].coef_[0]
print("Coefficients (on standardized features):", dict(zip(ML_FEATURES, coefs.round(3))))

# Confusion matrix = 2x2 table of (real answer) x (predicted answer):
#   TN = normal predicted normal (correct)   FP = normal predicted viral (false alarm)
#   FN = viral predicted normal (missed!)    TP = viral predicted viral (correct)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()  # .ravel() flattens the 2x2 table into 4 numbers
print(f"\nConfusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")

# Draw the confusion matrix as a coloured picture (darker = more videos)
disp = ConfusionMatrixDisplay(cm, display_labels=["normal", "viral"])
disp.plot(cmap="Blues")
plt.title("Logistic Regression - confusion matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=100)
plt.close()
print("Plot saved as 'confusion_matrix.png'")

# Results table: every test video + real answer, prediction, probability and outcome
ml = test_df[["id", "video_title"] + ML_FEATURES + ["views_count"]].copy()
ml["true_label"] = y_test.values
ml["predicted"] = y_pred
ml["probability_viral"] = y_prob.round(3)
# np.select(list_of_conditions, list_of_values, default): like an if/elif/else for
# a whole column - the first condition that is True decides the value
ml["outcome"] = np.select(
    [(ml["true_label"] == 1) & (ml["predicted"] == 1),
     (ml["true_label"] == 0) & (ml["predicted"] == 0),
     (ml["true_label"] == 0) & (ml["predicted"] == 1)],
    ["TP", "TN", "FP"], default="FN",
)
print("\nMisclassified objects:")
# .isin([...]) keeps rows whose value is in the list -> only the mistakes
print(ml[ml["outcome"].isin(["FP", "FN"])][
    ["id", "video_title", "day_since_published", "views_count", "true_label", "predicted", "probability_viral", "outcome"]
].to_string(index=False))

ml.to_csv("ml_detection.csv", index=False)
print("Saved: ml_detection.csv")

# 20 test rows is a tiny sample: one video changes accuracy by 5 points.
# 5-fold cross-validation on all 100 videos gives a more stable estimate:
# the data is cut into 5 parts of 20; the model is trained 5 times, each time
# tested on a different part and trained on the other 4. We report the average
# (mean) and how much the 5 results differ (std, the "+/-").
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
print("\n5-fold cross-validation on all 100 videos (more stable than one 20-row test):")
for name, model in [("Logistic Regression", lr), ("Random Forest (Task 1)", rf)]:
    acc = cross_val_score(model, df[ML_FEATURES], df[TARGET], cv=cv, scoring="accuracy")
    f1 = cross_val_score(model, df[ML_FEATURES], df[TARGET], cv=cv, scoring="f1")
    print(f"  {name:<23} accuracy = {acc.mean():.3f} +/- {acc.std():.3f} | F1 = {f1.mean():.3f} +/- {f1.std():.3f}")

# Leakage demonstration: the same models WITH views_count.
# Because label = (views_count > 100 000), these scores are almost perfect -
# which proves why views_count had to be excluded from ML_FEATURES above.
leak_features = ML_FEATURES + ["views_count"]
leak = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
leak.fit(train_df[leak_features], y_train)
leak_pred = leak.predict(test_df[leak_features])
leak_rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE).fit(train_df[leak_features], y_train)
print("\nFor comparison - models that also see views_count (label = views > 100K, i.e. leakage):")
print(f"  Logistic Regression F1 = {f1_score(y_test, leak_pred):.4f}")
print(f"  Random Forest       F1 = {f1_score(y_test, leak_rf.predict(test_df[leak_features])):.4f}")
