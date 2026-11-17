# Lab 7: What Was Done and How the Code Works

*Written for someone who did Labs 4 and 6 (classification and anomaly detection) and is new to "IDS-style" detection pipelines.*

## Overview

Lab 7 asks for a small **cyber threat detection system** with 4 parts:

| Task | What the lab asks | What it became for our dataset |
|---|---|---|
| 1 | IDS with Decision Tree / RF / SVM + alerts INFO/WARNING/CRITICAL | Random Forest that flags **likely-viral videos** among 20 "new events" |
| 2 | Risk levels LOW/MEDIUM/HIGH, TOP-5, chart, response measures | Investigation priority for the flagged videos + response playbook |
| 3 | Rule-based detection from 3 features (malware example) | Hand-written rule: volume + velocity of views + clickbait title → NORMAL/SUSPICIOUS |
| 4 | ML classification with metrics + confusion matrix (phishing/TF-IDF example) | Logistic Regression with Accuracy/Precision/Recall/F1, confusion matrix, error analysis |

**Dataset:** the same `clean_dataset.csv` (100 self-improvement YouTube videos, Labs 2–6).
**Target:** `label` (True = viral, False = normal).

### How "viral video" maps onto a cybersecurity threat

The PDF says the malware and phishing parts are only examples, and that the methods should be applied to *your own* topic. We treat a viral video as an **abnormal-popularity event**. That is the moment a video becomes a security target:

- **view-botting / fake engagement:** are the views real or bought?
- **scam & spam comment bots:** viral comment sections attract crypto scams and phishing links
- **impersonation channels:** fake channels re-upload the video with "giveaway" scams
- **account takeover:** a channel that suddenly blows up is a valuable account to steal

So "detect viral videos early and rank them" is like an IDS that detects a traffic spike and decides which ones an analyst should look at first.

### Files in this folder

| File | What it is |
|---|---|
| `lab_7.py` | The lab: all 4 tasks in one script |
| `Lab7_Code.ipynb` | Same code split into notebook cells, with beginner explanations (required deliverable) |
| `make_screenshots.py` | Runs `lab_7.py` and renders console output as terminal-style PNGs (one per task) |
| `clean_dataset.csv` | Copy of the Lab 3 dataset |
| `event_log.csv` | Task 1 output: 20 test events with prediction, risk_score and alert level |
| `risk_analysis.csv` | Task 2 output: event log + `risk_level` + `response` |
| `threat_detection.csv` | Task 3 output: all 100 videos with scaled indicators, risk_score, NORMAL/SUSPICIOUS |
| `ml_detection.csv` | Task 4 output: 20 test videos with prediction, probability and TP/TN/FP/FN |
| `risk_distribution.png` | Task 2 bar chart |
| `confusion_matrix.png` | Task 4 confusion matrix |
| `screenshots/` | Execution screenshots for the report |
| `Lab7_Report.md` | Report with placeholders for name/group |

**How to re-run** (from `src/lab7`):
```bash
python lab_7.py            # all tasks -> 4 CSVs + 2 plots
python make_screenshots.py # screenshots/
```

---

## The most important decision: `views_count` is NOT given to the ML models

While checking the data, I found that `label` is **exactly** `views_count > 100 000`:

- every `label = False` video has ≤ 95 547 views
- every `label = True` video has ≥ 104 585 views

So if a model sees `views_count`, it just learns "views > ~100K" and gets 100%. That is **data leakage**: the answer is hidden inside the input. You can see it at the end of the script's output:

```
For comparison - models that also see views_count (label = views > 100K, i.e. leakage):
  Logistic Regression F1 = 0.9000
  Random Forest       F1 = 1.0000
```

(In Lab 4, `views_count` had ~0.82 importance in the Random Forest for this reason.)

To make the detection meaningful, the ML parts (Tasks 1 and 4) use only signals that are known **before the views pile up**. That turns it into an *early-warning* detector:

```python
ML_FEATURES = ["title_length", "contains_number", "has_uppercase_words", "day_since_published"]
```

`views_per_day` is also excluded, because `views_per_day × day_since_published = views_count`, and a tree would rebuild the leak.

The cost is honest but low scores: these 4 features carry only a weak signal (cross-validated accuracy ≈ 0.58–0.62, where 0.50 is random). That weakness **is the finding**: a title's length or style alone does not reliably predict virality.

> To change this: add `"views_count"` to `ML_FEATURES` at the top of `lab_7.py` and re-run. Everything still works, but Tasks 1 and 4 become trivial (≈100%).

Task 3 (rules) *does* use views, because there the views are the observed "activity" being inspected, like traffic volume in an IDS. It is not a learned model, so leakage doesn't apply in the same way.

---

## Setup block (top of the script)

```python
df = pd.read_csv("clean_dataset.csv")
for col in ["contains_number", "has_uppercase_words", "label"]:
    df[col] = df[col].astype(int)          # True/False -> 1/0
df["views_per_day"] = (df["views_count"] / df["day_since_published"].clip(lower=1)).round(1)
```

- Boolean columns become 0/1 so sklearn and arithmetic work on them.
- `views_per_day` is a new **derived feature**: how fast the video gains views. `.clip(lower=1)` avoids dividing by 0 for videos uploaded today.

```python
train_df, test_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df[TARGET])
```

One split is shared by Tasks 1 and 4: 80 videos for training and 20 test videos. The 20 test videos are the "**new events**" the lab asks for ("at least 20 test events"). `stratify` keeps 10 viral and 10 normal videos in the test set.

---

## Task 1: Detection system (Random Forest + alerts)

```python
rf = RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=2, random_state=42)
rf.fit(train_df[ML_FEATURES], train_df[TARGET])
predictions   = rf.predict(test_df[ML_FEATURES])            # 0 / 1
probabilities = rf.predict_proba(test_df[ML_FEATURES])[:, 1] # P(viral)
```

- **Random Forest** = 200 decision trees, each trained on a random sample of rows and features; the final answer is a vote. `predict_proba` is the share of trees voting "viral".
- `max_depth=4`, `min_samples_leaf=2` keep trees shallow. With only 80 training rows, deep trees would memorise noise (overfitting).

**Alert levels:**
```python
def alert_level(prediction, probability):
    if prediction == 0: return "INFO"        # looks normal
    if probability >= 0.60: return "CRITICAL" # clearly predicted viral
    return "WARNING"                          # predicted viral, but unsure
```

The lab example uses 0.85, but our model never gets that confident: its probabilities stay between about 0.26 and 0.68, because the features are weak. With 0.85, CRITICAL would never fire. 0.60 is the point where the model is clearly on the "viral" side of 0.5.

**Result:** 4 CRITICAL, 8 WARNING, 8 INFO. Of the 12 videos flagged as viral, 7 really are viral (precision ≈ 0.58), and 7 of the 10 viral videos were caught (recall 0.70). Feature importance: `day_since_published` (0.48) and `title_length` (0.39) carry most of the signal. Older videos had more time to collect views.

Saved to `event_log.csv`. The file also keeps `views_count` / `true_label` **for analysis only**. They are not model inputs.

---

## Task 2: Risk assessment

Reads `event_log.csv` back (as the lab requires) and assigns a priority:

| Level | Rule | Why |
|---|---|---|
| HIGH | predicted viral **and** p ≥ 0.60 | same as CRITICAL: strongest evidence |
| MEDIUM | predicted viral, **or** p ≥ 0.45, **or** (video ≤ 30 days old **and** p ≥ 0.30) | borderline cases near 0.5; fresh videos are where a spike is still developing, so a lower bar applies |
| LOW | everything else | model is fairly sure it's an ordinary video |

**TOP-5** is sorted by risk level, then probability, then *fresher first* (a newer spike is more urgent).

**Response playbook** is added as a `response` column:

- HIGH → verify view authenticity (bot traffic), turn on comment spam/scam filtering, alert the channel owner, enforce 2FA
- MEDIUM → watch-list, re-score in 24 h, sample comments for scam links
- LOW → routine monitoring

**Result:** LOW 4, MEDIUM 12, HIGH 4 → `risk_distribution.png`, `risk_analysis.csv`.
Three of the four HIGH videos really are viral. The fourth (id 79, "Why You REALLY Procrastinate…") is a false alarm: it is old (926 days) and has a CAPS word, which the model associates with virality.

---

## Task 3: Rule-based detection (no ML)

This mirrors the malware example ("mass file changes + encryption + odd behaviour → sum → threshold").

**Three indicators:**

| Indicator | Column | IDS analogy |
|---|---|---|
| Volume | `views_count` | traffic volume |
| Velocity | `views_per_day` | requests per second |
| Lure | `contains_number` OR `has_uppercase_words` (clickbait title) | phishing bait |

**Scaling (required by the lab note):** views range from 182 to 11.4 M. Plain min-max scaling would push almost everything to ~0 because of one mega-viral video, so we first take a log and then min-max:
```python
def log_minmax(series):
    s = np.log1p(series)                       # log(1 + x): compresses huge values
    return (s - s.min()) / (s.max() - s.min()) # -> 0..1
```

**Score:** weighted sum in 0..1:
```python
risk_score = 0.45*volume_score + 0.45*velocity_score + 0.10*clickbait_score
```
The title is only a weak hint (as Task 1 showed), so it gets a small weight.

**Threshold choice:** the script prints a sweep of thresholds 0.40–0.70. 0.60 was chosen because it is the last point where the rule still catches > 90% of viral videos while keeping false alarms low. Higher thresholds miss too many; lower ones flag 2/3 of the dataset. `label` is printed **only as a reference** for picking a threshold; the rule itself never reads it.

**Result:** 49 SUSPICIOUS / 51 NORMAL; 93% agreement with `label`. This is expected, because the rule looks at views and the label is defined by views. → `threat_detection.csv`.

---

## Task 4: ML classification + evaluation

The phishing example uses **TF-IDF** (turns text into numbers). The lab says TF-IDF is mandatory only when there is text to classify. Our features are numeric, so we use **StandardScaler** instead (mean 0, std 1):

```python
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
lr.fit(X_train, y_train)
```

`make_pipeline` guarantees the scaler is fitted **only on the training data** and then reused on the test data. Fitting on test data would leak information.

**Metrics on the 20 test videos:**

| Accuracy | Precision | Recall | F1 |
|---|---|---|---|
| 0.45 | 0.40 | 0.20 | 0.27 |

Confusion matrix: TN = 7, FP = 3, FN = 8, TP = 2. The model mostly says "normal" and misses 8 of the 10 viral videos.

**Why so low, and is it just bad luck?** 20 rows is tiny: one video moves accuracy by 5 points. So the script also runs **5-fold cross-validation** on all 100 videos (train on 80, test on 20, five times with different splits):

| Model | CV accuracy | CV F1 |
|---|---|---|
| Logistic Regression | 0.58 ± 0.05 | 0.49 ± 0.12 |
| Random Forest (Task 1) | 0.62 ± 0.08 | 0.59 ± 0.12 |

So this particular split was unlucky for LR. Even on average, though, the pre-view features are only slightly better than a coin flip. Random Forest is better because it can learn non-linear rules such as "old **and** short title".

**Error analysis** (printed as "Misclassified objects", column `outcome` in `ml_detection.csv`):

- **False negatives (8):** viral videos the model missed, e.g. id 49 (2.5 M views, 80 days old) and id 9 (552 K views, 20 days old). In security terms, a missed attack: no protection is turned on while scam bots move into the comments. **This is the more dangerous error.**
- **False positives (3):** ids 84, 98, 79, normal videos flagged as viral. For example, id 98 is the oldest video (1945 days) but has only 2.7 K views; the model over-trusts "old ⇒ viral". In security terms, a false alarm: wasted analyst time and alert fatigue.

---

## Helper scripts

- **`make_screenshots.py`** is a copy of the Lab 6 one with `lab_6.py` → `lab_7.py`. It runs the lab, splits the output at the `=== Task N` headers and draws each part as a PowerShell-style PNG.
- **`Lab7_Code.ipynb`** contains the same code as `lab_7.py`, one cell per task, with a plain-language intro above each cell and beginner comments in the code. If you change `lab_7.py`, copy the change into the matching notebook cell (and the other way round).
