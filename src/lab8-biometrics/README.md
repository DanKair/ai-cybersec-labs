# Lab 8 — AI-Based Biometric Identification System

**Face, Iris and Fingerprint Recognition** · Course: *Artificial Intelligence in Cybersecurity* · Google Colab / Python

This repository is our team's implementation of Laboratory Work No. 8. A single Jupyter notebook
(`Lab8_Biometrics.ipynb`) covers all four assignments plus the comparative analysis, and runs
both on Google Colab and locally.

> **Status of the data in this repo**
> * `data/student_faces/<Name>/` folders are **intentionally empty**. Each team member puts their own 5 photos there.
> * `data/iris_dataset/` and `data/fingerprint_dataset/` already contain the open-dataset subsets used for Assignments 3–4.
> * All numbers, CSVs and figures currently in `results/`, `*_results.csv` and the notebook outputs come from a
>   **demo run on a mock face dataset** (Faces94, public research volunteers). Re-run the notebook with the real team
>   photos before the defense — every table and figure regenerates automatically.

---

## 1. Project overview

| # | Assignment | Method used | Data | Output |
|---|---|---|---|---|
| 1 | Custom facial image dataset | Folder/quality checks (resolution, RGB, one face) | Team photos | `results/face/dataset_*.png`, `dataset_quality.csv` |
| 2 | Interactive face recognition | OpenCV face detector → **FaceNet512** embeddings → cosine distance → name / `UNKNOWN`; **Gradio** UI | Assignment 1 dataset | `face_results.csv`, annotated group photos |
| 3 | Iris recognition | CLAHE → pupil/iris circle segmentation → Daugman rubber-sheet → **IrisCode** (Gabor) + Hamming distance; ORB for comparison | CASIA-IrisV1 subset | `iris_results.csv`, segmentation figures |
| 4 | Fingerprint recognition | Gaussian blur → CLAHE → Otsu → skeleton + minutiae; **ORB + RANSAC** matching | SOCOFing subset | `fingerprint_results.csv`, stress test |
| 5 | Comparative analysis | Table + chart built from the computed metrics | Results of 2–4 | `results/comparison/` |

### Demo-run results (mock faces + open iris/fingerprint data)

| Modality | Mandatory 10-pair test / group photos | Extended test | Notes |
|---|---|---|---|
| Face | 5 group photos: **20/20 registered faces identified**, 3/3 strangers → `UNKNOWN` | `test_05`: 4/4 | Mock data is easy (same session); expect lower on real photos |
| Iris (IrisCode) | **9/10 correct** (1 cross-session false reject) | 105 pairs: 98.1 % acc, FAR 0 %, FRR 13.3 % | Error caused by an imperfect iris-circle fit |
| Fingerprint (ORB + RANSAC) | **10/10 correct** | 105 pairs: 100 %; stress test: extreme distortion → 40 % accepted | Variants are synthetic → optimistic |

---

## 2. About this lab work

**Goal:** study the main biometric identification techniques and build software prototypes that
recognise a person from a face photo, an iris image and a fingerprint, then evaluate them experimentally
and analyse their errors and security limitations.

Key concepts the defense will ask about:

| Term | Meaning | In this project |
|---|---|---|
| **Detection** | *Where* is the biometric object in the image? | Face boxes, pupil/iris circles |
| **Verification (1:1)** | Are these two samples the same person? | Iris & fingerprint pairs → `MATCH` / `NO MATCH` |
| **Identification (1:N)** | Who is this, searching the whole database? | Face app → student name or `UNKNOWN` |
| **Threshold** | Score limit that turns a similarity score into a decision | 0.30 cosine (face), 0.32 Hamming (iris), 12 inliers (fingerprint) |
| **FAR / FRR** | False Acceptance Rate (stranger accepted) / False Rejection Rate (owner rejected) | Computed in every section |

Assessment (100 pts): Dataset 15 · Face recognition 30 · Iris 15 · Fingerprint 15 · Analysis & report 10 ·
Live demo 10 · Individual defense 5. **Every team member must be able to explain every assignment** —
read [`EXPLANATION.md`](EXPLANATION.md) for a guided walkthrough and answers to the 15 review questions.

---

## 3. Project structure

```
lab8-biometrics/
├── README.md                    ← you are here: overview + setup
├── EXPLANATION.md               ← how the code works, step by step + defense Q&A
├── Lab8_Biometrics.ipynb        ← THE notebook: Assignments 1-4 + comparative analysis
├── requirements.txt             ← packages for a local run
├── .gitignore                   ← keeps real student photos / templates out of git
│
├── face_results.csv             ← Assignment 2 results (one row per group photo)
├── iris_results.csv             ← Assignment 3: 5 genuine + 5 impostor pairs
├── fingerprint_results.csv      ← Assignment 4: 5 genuine + 5 impostor pairs
│
├── data/
│   ├── student_faces/           ← Assignment 1 dataset (EMPTY — add your photos)
│   │   ├── Aidana/   ├── Alikhan/   ├── Aruzhan/   └── Daniyar/      (rename to real names)
│   │       photo_01.jpg  photo_02.jpg  photo_03.jpg  photo_04.jpg  test_05.jpg
│   ├── group_photos/            ← optional: group photos + ground_truth.csv for batch evaluation
│   │   └── ground_truth.example.csv
│   ├── iris_dataset/            ← CASIA-IrisV1 subset: iris_01..05 / image_01..03.jpg
│   └── fingerprint_dataset/     ← SOCOFing subset: finger_01..05 / image_01..03.bmp
│
├── results/                     ← every figure and table the notebook produces
│   ├── face/                    ← dataset sheets, threshold analysis, annotated group photos
│   ├── iris/                    ← preprocessing, segmentation, crops/ (iris_crops), score plots
│   ├── fingerprint/             ← preprocessing + minutiae, matches, stress test
│   ├── comparison/              ← comparative_analysis.csv/.md, accuracy chart
│   └── screenshots/             ← Gradio app screenshots
│
├── scripts/
│   ├── prepare_open_datasets.py ← re-downloads the iris + fingerprint subsets
│   └── make_demo_faces.py       ← builds the mock face dataset used for testing
│
└── report/
    ├── final_report.md          ← team report (16 required sections) with explained screenshots
    └── Lab8_Report.pdf          ← PDF version of the report
```

---

## 4. Setup guide

### 4.1 Prepare your photos (Assignment 1) — every team member

1. Rename the folders in `data/student_faces/` to the real team members' names (Latin letters, no spaces),
   and add/remove folders so there is exactly one per member.
2. Each member puts **exactly these 5 files** in their folder:

   | File | Photo | Used for |
   |---|---|---|
   | `photo_01.jpg` | Frontal, looking at the camera | Registration |
   | `photo_02.jpg` | Head turned ~20–30° to the right | Registration |
   | `photo_03.jpg` | Head turned ~20–30° to the left | Registration |
   | `photo_04.jpg` | Different lighting | Registration |
   | `test_05.jpg` | A new, separate photo | **Testing only** |

   Requirements: JPG/JPEG, ≥ 640×480, RGB, **one face only**, sharp, no filters/masks, neutral background.
   The notebook's quality check (section 1.3) flags any photo that breaks these rules.
3. Take **at least 5 group photos** (4–5 people each, different lighting/arrangements/angles) for the live defense.
   They must not be used for registration.

### 4.2 Run on Google Colab (required by the lab)

**Option A — upload ZIPs (the lab's way)**

1. Zip the folder: right-click `student_faces` → *Compress* → `student_faces.zip`
   (do the same for `iris_dataset` and `fingerprint_dataset` if you don't upload the whole project).
2. Open <https://colab.research.google.com> → *File → Upload notebook* → `Lab8_Biometrics.ipynb`.
3. *Runtime → Run all*. When a dataset is missing, a file picker appears — select the matching ZIP.
   The loader accepts a ZIP that contains the folder itself or only its sub-folders, ignores `__MACOSX`,
   and refuses unsafe paths (zip-slip protection).

**Option B — whole project from Google Drive (recommended, nothing to re-upload)**

1. Upload the entire `lab8-biometrics/` folder to *My Drive*.
2. Open the notebook from Drive and add/run this cell first:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/drive/MyDrive/lab8-biometrics
   ```
3. *Runtime → Run all*. Results are written back into the Drive folder.

> First run downloads the FaceNet512 weights (~95 MB) — allow 1–2 minutes.
> The Gradio app appears inside the output of cell **2.8**. Don't set `share=True`: it would publish your biometric system on a public URL.

### 4.3 Run locally

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook Lab8_Biometrics.ipynb                   # then Run All
```
The Gradio app opens at <http://127.0.0.1:7860>. To execute the notebook head-less (no UI), e.g. to regenerate results:
```bash
LAB8_LAUNCH_UI=0 jupyter nbconvert --to notebook --execute --inplace Lab8_Biometrics.ipynb
```

### 4.4 Optional helpers

```bash
python scripts/prepare_open_datasets.py        # re-create data/iris_dataset + data/fingerprint_dataset
python scripts/make_demo_faces.py --out data   # fill data/ with the MOCK faces + 5 synthetic group photos
```
Use the mock dataset only to test the pipeline. Delete it (`data/student_faces/*/*`, `data/group_photos/*`) before adding real photos.

### 4.5 Batch-evaluating your own group photos (optional, fills `face_results.csv`)

Put the group photos in `data/group_photos/` and create `ground_truth.csv` (copy `ground_truth.example.csv`):

```csv
photo,actual_people,registered_names,unregistered_people
group_01.jpg,4,Aidana;Alikhan;Aruzhan;Daniyar,0
group_02.jpg,5,Daniyar;Aruzhan;Alikhan;Aidana,1
```
During the live defense you can instead use the app's **Evaluation** panel — it appends each photo's
result to `face_results.csv` without touching the code.

### 4.6 Tuning without touching the logic

All parameters live in notebook cell **0.3**: `MODEL_NAME`, `DETECTOR` (`opencv` → `retinaface`/`mtcnn` for
harder photos), `FACE_THRESHOLD`, `IRIS_HD_THRESHOLD`, `FP_INLIER_THRESHOLD`, etc. Section 2.5 prints the
genuine/impostor distance ranges for your real photos — use it to justify your final threshold.

---

## 5. Live defense checklist

1. Open the notebook in Colab → *Run all* → show section 1 (folder tree, photo counts, quality table, photo sheets).
2. Scroll to cell 2.8 → show **Registered students: N**.
3. **Upload Photo** → **RECOGNIZE FACES** for each of the ≥ 5 new group photos; explain any `UNKNOWN`/errors
   (distance values are printed next to each name).
4. Show sections 3 and 4: dataset loading, preprocessing figures, `MATCH`/`NO MATCH` tables.
5. Show section 5 comparison table. Each assignment is presented by a different team member.

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `No faces detected` on a group photo | Faces too small/dark → use a closer photo or set `DETECTOR = "retinaface"` |
| A registered student comes out `UNKNOWN` | Check section 2.5 distances; raise `FACE_THRESHOLD` slightly (e.g. 0.35) or retake photo_04 |
| Wrong name assigned | Threshold too high, or a registration photo contains another face → check quality table |
| Phone photo appears rotated | Handled by `ImageOps.exif_transpose`; if not, re-save the photo |
| TensorFlow/deepface version error on Colab | *Runtime → Restart session* after the `pip install` cell, then *Run all* |
| Haar cascade download error | `pip install opencv-python` (4.x includes the cascade files), or retry with internet access |

## 7. Data sources, licences and ethics

* **Iris:** CASIA-IrisV1 (Chinese Academy of Sciences, Institute of Automation), via the GitHub mirror
  `yj7082126/OpenCV_IrisRecognition`. Academic, non-commercial use.
* **Fingerprint:** SOCOFing — Sokoto Coventry Fingerprint Dataset (Shehu et al., 2018), Kaggle `ruizgara/socofing`,
  CC BY-NC-SA 4.0, via the GitHub mirror `angeliquebouchard/SOCOfing`.
* **Mock faces (testing only):** Faces94, Dr Libor Spacek, University of Essex, via `chenyijia1997/Face-datasets`.
* Used **only for learning / academic defense**, not for commercial purposes.
* Team photos are collected with voluntary informed consent, are not committed to public repositories (see `.gitignore`),
  and face templates are kept in memory only.
