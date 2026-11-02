# Lab 6: Anomaly Detection Explained

*Explained for someone who knows pandas & numpy and did Lab 4 (classification), but is new to anomaly detection.*

## Overview

Lab 4 was **supervised**: we had `label` and taught models to predict it.
Lab 6 is **unsupervised**: we **ignore `label`** and ask the data itself "which rows look unusual compared to the rest?".

| | Lab 4 (classification) | Lab 6 (anomaly detection) |
|---|---|---|
| Needs a target `y`? | Yes (`label`) | No |
| Train/test split? | Yes | No – we fit on all rows |
| Output | 0/1 class for each row | "normal" / "anomaly" for each row |
| Can we compute F1? | Yes | Only if we have *verified* anomaly labels (we don't) |

**Dataset:** same `clean_dataset.csv` from Lab 3 (100 self-improvement YouTube videos).
**Algorithms:** Isolation Forest (IF) and Local Outlier Factor (LOF).

### Files in this folder

| File | What it is |
|---|---|
| `lab_6.py` | The lab itself – all 9 tasks |
| `make_screenshots.py` | Runs `lab_6.py` and renders its console output into terminal-style PNGs |
| `clean_dataset.csv` | Copy of the Lab 3 dataset |
| `anomaly_results.csv` | Dataset + 4 new columns (`IF_anomaly`, `IF_score`, `LOF_anomaly`, `LOF_score`) |
| `if_anomalies.png`, `lof_anomalies.png` | The two required plots (Task 7) |
| `anomalies_log_scale.png` | Extra plot – same thing with a log x-axis (easier to read) |
| `screenshots/taskN_output.png` | "Screenshots of execution" for the report, one per task |
| `lab_6_report.md` | The report with placeholders for your name/group |

**How to re-run** (from `src/lab6`):
```bash
python lab_6.py            # computes everything, saves CSV + plots
python make_screenshots.py # re-runs lab_6.py and regenerates screenshots/
```

---

## Task 1: Load Data

```python
df = pd.read_csv("clean_dataset.csv")
```
Same as every lab. `Shape: (100, 9)` → 100 videos, 9 columns. I hide `video_url` in the printout only because it's long.

---

## Task 2: Select Features

```python
features = ["views_count", "day_since_published", "title_length"]
```

**Why these three?** They are the only *continuous* numeric columns.
- `contains_number`, `has_uppercase_words` are True/False. Lab allows encoding them, but 0/1 columns create many identical distances, which confuses LOF (it's distance-based). So I left them out.
- `id` is just a row number – meaningless.
- `label` must **not** be a feature – this is unsupervised.

**Order matters only for the plot:** Task 7 draws `features[0]` vs `features[1]`, so `views_count` (x) vs `day_since_published` (y).

`X.apply(pd.to_numeric, errors="raise")` – crashes if any value isn't a number (safety check).
Then two checks: no NaN, at least 10 rows. Both pass.

---

## Task 3: Standardize

```python
X_scaled = StandardScaler().fit_transform(X)
```
Formula: `z = (x - mean) / std`.

**Why it matters a LOT here:** `views_count` goes up to 11 million, `title_length` up to 82. Both IF and LOF (especially LOF, which measures distances) would otherwise basically only "see" views_count. After scaling every column has mean 0 and std 1 → each feature has equal voice.

Note: unlike Lab 4, we fit the scaler on **all** data – there's no test set to "leak" into.

---

## Task 4: Isolation Forest

```python
iso = IsolationForest(contamination=0.10, random_state=42)
iso_pred = iso.fit_predict(X_scaled)
```

**Idea in one sentence:** randomly cut the data with vertical/horizontal lines; a weird point gets "isolated" (alone in its box) after very few cuts, a normal point buried in a crowd needs many cuts.

- Builds 100 random trees. For each row, average how deep it is in the trees (path length). Short path = anomaly.
- `fit_predict` returns **-1 = anomaly, +1 = normal** (sklearn convention). We convert: `(iso_pred == -1).astype(int)` → **1 = anomaly, 0 = normal**.
- `IF_score = -iso.score_samples(...)` – sklearn's score is "higher = more normal", so we flip the sign → **higher = more anomalous**.
- `contamination=0.10` means "I *assume* 10% of rows are anomalies" → the model marks exactly the top 10 most isolated rows. This is an assumption we picked, **not** something the data told us.

**Result:** 10 anomalies (10%).

---

## Task 5: Local Outlier Factor

```python
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.10)
```

**Idea:** for each point look at its 20 nearest neighbours. Compare "how crowded is my neighbourhood" vs "how crowded are my neighbours' neighbourhoods".
- LOF ≈ 1 → I'm as dense as my neighbours → normal.
- LOF ≫ 1 → I'm in a much emptier area than my neighbours → anomaly.

`LOF_score = -lof.negative_outlier_factor_` → the actual LOF value (e.g. 4.8 for video #1).

**Result:** 10 anomalies (10%) – again because contamination forces 10%.

---

## Task 6: Compare

Both found 10 – that's *forced* by contamination, so the count itself tells us nothing. What matters is **are they the same 10 videos?**

```
LOF   0  1
IF
0    87  3
1     3  7
```
- 7 videos flagged by both
- 3 only IF, 3 only LOF

**ARI (Adjusted Rand Index) = 0.617**
- ARI compares two groupings: 1 = identical, 0 = no better than random, can go negative.
- 0.617 = "fairly similar but not the same".
- ⚠️ ARI only says the two algorithms *agree*. It does **not** say either is *correct* – there's no ground truth.

### Who are the anomalies? (my interpretation)

| Group | Videos (id) | Why |
|---|---|---|
| Both | 1, 13, 96, 62 | Viral: 4.6M–11.5M views (dataset median is ~100K) |
| Both | 98 | Opposite: oldest video (1945 days) but only 2,699 views |
| Both | 77, 68 | Mix of high views and/or old age |
| Only IF | 100, 11, 83 | Large single values (3.5M views; 82-char title – the longest) |
| Only LOF | 67, 69, 64 | Old (1100–1500 days) with moderate views – sit in a sparse area of the plot |

**Why do they differ?**
- **IF** is *global*: it likes points that are extreme on *any* single feature (very high views, longest title).
- **LOF** is *local*: it likes points that are in an *empty region* compared to their neighbours. Old videos (>1000 days) are rare, so they live in an empty area, even if their views are normal.

---

## Task 7: Plots

Two scatter plots (x = views_count, y = day_since_published), red = anomaly, blue = normal.

Problem: views_count is extremely skewed → on a linear axis almost all points are squashed against x=0. So I added `anomalies_log_scale.png` – same data, log x-axis, much clearer. The model itself still used linear (scaled) values; the log is **only for display**.

**Limitation:** the model used 3 features, the plot shows 2. A red point that looks "normal" on the plot is probably weird in `title_length` (e.g. #11 with 82 chars). Never judge an anomaly only by a 2D picture.

---

## Task 8: Is `label` the ground truth?

**No.** In Lab 2 `label` was assigned by hand as a binary classification target, split 50/50. Anomalies are by definition *rare*; a 50/50 label can't be "anomaly yes/no". So computing Precision/Recall/F1 against it would be meaningless – the code keeps those lines commented out, as the lab says.

The crosstab is printed for info only: 9 of 10 IF anomalies have `label=True`. That's not "IF is accurate" – it's because `label` correlates with `views_count` (Lab 4 found views_count importance ≈ 0.82) and IF anomalies are mostly high-view videos.

---

## Task 9: Save

`anomaly_results.csv` = original 9 columns + `IF_anomaly`, `IF_score`, `LOF_anomaly`, `LOF_score`. Submit it with the report.

---

## How the screenshots were made

The lab asks for "personal screenshots of execution". `make_screenshots.py`:
1. runs `lab_6.py` via `subprocess` and captures its printed output,
2. splits the output by the `=== Task N` headers,
3. draws each part as dark-terminal-style PNG with matplotlib (`screenshots/task1_output.png` … `task9_output.png`, plus `full_run_output.png`).

If your teacher wants real screenshots, just run `python lab_6.py` in your terminal and press Win+Shift+S. 🙂

---

## Cheat sheet for the review questions

| Q | Short answer |
|---|---|
| Anomaly? | A row that differs a lot from typical behaviour |
| vs classification? | No target needed; finds "unusual", not "class A/B" |
| Anomaly ≠ attack? | Unusual can be legit (viral video, admin running a backup at 3am) |
| IF? | Random splits; anomalies isolated in fewer splits |
| LOF? | Compares local density to neighbours' density |
| -1 / +1? | -1 = outlier, +1 = inlier |
| Standardize? | So large-scale features don't dominate distances/splits |
| contamination? | Assumed share of anomalies → sets the threshold |
| Why IF ≠ LOF? | Global isolation vs local density |
| ARI? | Agreement between two partitions, not correctness |
| label as truth? | No – 50/50 manual classification label |
| 2D plot limits? | Hides other features; can't see why a point is anomalous |
