# EXPLANATION — How the Lab 8 code works

This document explains `Lab8_Biometrics.ipynb` section by section: **what** each part does, **why** it is
done that way, and **what to say** when the teacher asks about it. Read it once top-to-bottom; every team
member should be able to explain all four assignments.

Contents
1. [The big picture](#1-the-big-picture)
2. [Section 0 — Setup and configuration](#2-section-0--setup-and-configuration)
3. [Assignment 1 — Face dataset](#3-assignment-1--face-dataset)
4. [Assignment 2 — Face recognition system](#4-assignment-2--face-recognition-system)
5. [Assignment 3 — Iris recognition](#5-assignment-3--iris-recognition)
6. [Assignment 4 — Fingerprint recognition](#6-assignment-4--fingerprint-recognition)
7. [Section 5 — Comparative analysis](#7-section-5--comparative-analysis)
8. [Where we improved on the lab's example code](#8-where-we-improved-on-the-labs-example-code)
9. [Helper scripts](#9-helper-scripts)
10. [Defense Q&A — the 15 review questions](#10-defense-qa--the-15-review-questions)

---

## 1. The big picture

Every biometric system in this lab follows the same 5-step skeleton. Only the contents of each step change:

```
 ┌──────────┐   ┌─────────────┐   ┌──────────────┐   ┌───────────────┐   ┌─────────────────┐
 │ 1. Load  │──►│ 2. Pre-     │──►│ 3. Detect /  │──►│ 4. Extract    │──►│ 5. Compare with │
 │  image   │   │  process    │   │  segment     │   │  features     │   │  threshold      │
 └──────────┘   └─────────────┘   └──────────────┘   └───────────────┘   └─────────────────┘
```

| Step | Face | Iris | Fingerprint |
|---|---|---|---|
| 2. Preprocess | RGB, EXIF rotation fix | Grayscale, CLAHE | Grayscale, upscale ×3, Gaussian blur, CLAHE |
| 3. Detect / segment | OpenCV face detector + alignment | Pupil circle (threshold), iris circle (Hough) | Otsu binarisation, finger mask |
| 4. Features | FaceNet512 → 512 numbers | Rubber-sheet strip → IrisCode bits | ORB keypoints (+ minutiae for analysis) |
| 5. Compare | Cosine distance ≤ 0.30 → name | Hamming distance < 0.32 → MATCH | RANSAC inliers ≥ 12 → MATCH |
| Mode | **Identification 1:N** | **Verification 1:1** | **Verification 1:1** |

**Score direction matters:** for face and iris the score is a *distance* (smaller = more similar);
for fingerprints it is a *count of matches* (bigger = more similar). The decision rule flips accordingly.

---

## 2. Section 0 — Setup and configuration

**0.1 Install.** On Colab only, `!pip install deepface gradio`. `IN_COLAB` is detected by checking
whether the `google.colab` module is loaded, so the same notebook runs locally without changes.

**0.2 Imports.** OpenCV (image operations), NumPy (vectors), pandas (tables → CSV), matplotlib (figures),
Pillow (`ImageOps.exif_transpose` rotates phone photos the way the phone displayed them), scikit-image
(`skeletonize` for fingerprints). Random seeds are fixed so experiments give the same numbers every run.

**0.3 Configuration.** Every tunable value is in one cell — paths, model name, detector, all thresholds.
During the defense you change *parameters*, never *logic*. This is the "single source of truth" principle.

**0.4 Dataset loader.**
* `ensure_dataset(folder)` — if the folder already has images, use it; otherwise (on Colab) open the
  upload dialog and extract the ZIP.
* `safe_extract()` — two practical problems solved:
  1. ZIPs made on Windows contain `student_faces/...`; some people zip only the sub-folders. Both work.
  2. **Zip-slip protection:** a malicious ZIP can contain a path like `../../etc/passwd`; extracting it
     blindly writes outside the target folder. We resolve every path and refuse anything that escapes.
     This is a real CVE class — relevant for a cybersecurity course.
* `load_rgb(path)` — always returns an RGB `uint8` array with the correct orientation.

---

## 3. Assignment 1 — Face dataset

| Cell | What it does |
|---|---|
| 1.1 | Loads/uploads `data/student_faces` (lab's Code Example 1). |
| 1.2 | Counts students and photos and prints the folder tree (Code Example 2). |
| 1.3 | **Quality check** of every required photo: exists? JPEG? RGB? ≥ 640×480? exactly **one** face? Saved as `results/face/dataset_quality.csv`. |
| 1.4 | Shows the 5 photos of every student (Code Example 3 extended to all students) → `results/face/dataset_<Name>.png`. |

Why the quality check matters: a registration photo with a second face in the background (a poster, a
friend) silently registers the wrong embedding. Catching it here prevents mysterious errors later.

Why `test_05.jpg` is never registered: if the test photo is in the database, the system "recognises"
a photo it has literally memorised — the accuracy would be fake. Same rule as train/test split in ML.

---

## 4. Assignment 2 — Face recognition system

### 4.1 The pipeline

```
group photo ─► extract_faces (detect + align) ─► for each face: FaceNet512 ─► 512-D vector
                                                                              │
     database = {"Aidana": [v1,v2,v3,v4], "Alikhan": [...], ...}  ◄── cosine distance to every vector
                                                                              │
                         best distance ≤ FACE_THRESHOLD ? best name : "UNKNOWN"
```

### 4.2 What an embedding is

FaceNet is a convolutional neural network trained on millions of faces with a *triplet loss*: it was
rewarded for putting two photos of the same person close together and photos of different people far
apart. Its output — 512 numbers — is called an **embedding**. We never train anything: we use the
pretrained network as a "face → coordinates" converter. Recognition then becomes geometry: *which
registered point is nearest to the new point?*

### 4.3 Key functions

**`detect_and_embed(image_rgb)`** — the single function used for *both* registration and recognition.
Using one function for both is deliberate: if registration and recognition preprocessed faces
differently (different crop, colour order, alignment), the same person would produce different vectors.

1. Convert RGB → BGR (OpenCV and DeepFace expect BGR arrays).
2. `DeepFace.extract_faces(..., align=True)` finds every face and rotates it so the eyes are horizontal.
   `enforce_detection=True` makes it raise an error when there is no face — we catch it and return `[]`
   (instead of the library silently returning the whole image as "a face").
3. Faces smaller than `MIN_FACE_SIZE` (40 px) are dropped — the Haar detector sometimes fires on
   textures like hair or clothing patterns.
4. Each aligned face (float RGB 0–1) → `uint8` → **BGR** → `DeepFace.represent(..., detector_backend="skip")`
   → 512-D vector. `"skip"` because the face is already cropped.

**Registration (2.3).** Only `photo_01..04` are processed (`assert filename != TEST_FILE` guards it).
If a registration photo contains several faces we keep the largest and print a warning.
Result: `database = {name: [4 vectors]}`.

**`cosine_distance(a, b)`** = `1 − (a·b)/(|a|·|b|)`. It compares the *direction* of two vectors, not their
length: 0 = same direction (same person), ~1 = unrelated. Example: if two vectors point at 25° to each
other, cos(25°) = 0.906 → distance 0.094 → very likely the same person.

**`recognize_embedding(query, database)`** — nearest neighbour over *all* registered vectors (4 per
student). Keeping all 4 instead of averaging them helps when the query matches one specific condition
(e.g. the side view `photo_02`). If even the best distance is above the threshold → `UNKNOWN`.

### 4.4 Choosing the threshold (2.5)

We measure two lists on our own data:
* **genuine distances** — each `test_05` vs the same student's registered vectors;
* **impostor distances** — each `test_05` vs every *other* student's vectors (this simulates a stranger).

Then for every threshold from 0.05 to 0.80 we compute **FAR** (impostors accepted) and **FRR** (genuine
rejected) and plot them (`results/face/threshold_analysis.png`).

Demo-run numbers: genuine ≤ 0.163, impostor ≥ 0.786 → any threshold between them separates perfectly;
we keep **0.30** (DeepFace's reference value for FaceNet512 + cosine). It sits closer to the genuine side
on purpose: in a security setting a false acceptance (wrong person gets in) is worse than a false
rejection (the person retries). With real phone photos genuine distances are typically 0.15–0.30, so
**re-check this plot with your own data** and adjust if a team member is rejected.

### 4.5 Group photo recognition (2.6) — `recognize_uploaded_photo(photo)`

Input: PIL image from Gradio. Output: annotated image + text summary.
For every detected face (sorted left → right): recognise, draw a **green** box (known) or **red** box
(`UNKNOWN`), and write `Name (distance)` on a filled label. Line width and font scale with the photo size,
so labels stay readable on 4000-px phone photos. The summary lists detected / recognized / unknown counts
and each face's distance — useful when explaining an error to the teacher.

### 4.6 Evaluation → `face_results.csv` (2.7)

`evaluate_prediction()` compares the predicted names with the ground-truth list *as a multiset*, so we don't
need to know which box is which person:

* **correct** = predicted names that are really in the photo (each counted once)
* **incorrect** = names of students who are not in the photo (or a name predicted twice)
* **unknown** = faces labelled `UNKNOWN` (correct if the person is unregistered)
* **missed_registered** = registered students present but not correctly named (not detected or `UNKNOWN`)
* **Accuracy** = Σ correct / Σ registered students present × 100 %

Columns: `photo, actual_people, registered_present, detected_faces, correct, incorrect, unknown,
missed_registered, accuracy_%, predicted_names, timestamp`.

Two ways rows are added: (a) batch mode, if `data/group_photos/ground_truth.csv` exists; (b) live mode,
from the app's *Evaluation* panel (`log_result()`).

### 4.7 The Gradio app (2.8)

`gr.Blocks` layout: title → registered count → **Upload Photo** | **Recognition Result** →
**RECOGNIZE FACES** button → *Identification Results* textbox → collapsible *Evaluation* panel.
`recognize_button.click(fn=recognize_uploaded_photo, inputs=..., outputs=[...])` wires the button to the
function. Gradio keeps the Python process (and the `database` in memory) alive, so any number of photos
can be uploaded without restarting or editing code — the lab's main requirement.

`LAB8_LAUNCH_UI=0` (environment variable) skips `demo.launch()` so the notebook can be executed
automatically to regenerate results.

---

## 5. Assignment 3 — Iris recognition

### 5.1 Data
CASIA-IrisV1 subset: 5 eyes × 3 images (320×280 grayscale, near-infrared). `image_01`, `image_02` are
from session 1, `image_03` from session 2 (weeks later). CASIA-V1 pupils were pre-filled with a uniform
dark circle by the dataset authors, which makes pupil detection easy.

### 5.2 Preprocessing (3.3)
Grayscale + **CLAHE** (Contrast Limited Adaptive Histogram Equalization): the image is split into 8×8
tiles, each tile's histogram is stretched, and the stretch is *limited* (`clipLimit=2.0`) so noise isn't
amplified. Result: faint iris furrows and crypts become visible.

### 5.3 Segmentation (3.4) — `segment_iris(gray)`
1. **Pupil:** median blur → threshold `< 60` (pupil = darkest region) → morphological opening removes
   eyelash specks → contours → keep the biggest *round* contour (circularity = area / (π r²) > 0.6) →
   `minEnclosingCircle` gives centre and radius.
2. **Iris (limbus):** `cv2.HoughCircles` searches for circles with radius 1.6–3.8 × pupil radius whose
   centre is within half a pupil radius of the pupil centre (pupil and iris are nearly concentric). If
   Hough finds nothing, fall back to 2.4 × pupil radius (a typical anatomical ratio).
3. **Crops:** the ring between the circles is cut out and saved to `results/iris/crops/` — this is the
   lab's `iris_crops` folder.

The red/green circle overlays (`iris_segmentation.png`) let you verify every segmentation visually.

### 5.4 Normalisation (3.5) — Daugman's rubber-sheet model
`normalize_iris()` samples the ring along 512 angles × 64 radii and lays it out as a 64×512 rectangle:
row 0 = pupil edge, row 63 = iris edge, column = angle. This cancels out pupil dilation (the ring gets
thinner/thicker but the strip is always 64 rows) and turns eye rotation into a simple left/right shift.

### 5.5 IrisCode (3.5) — `iris_code()` and `hamming_distance()`
* The strip is filtered with **Gabor filters** (a sine wave inside a Gaussian window) at two wavelengths
  (16 and 32 px), each with an even (cosine) and odd (sine) version → 4 filter responses.
* Only the **sign** of each response is kept → 4 × 64 × 512 bits. Signs are robust to brightness changes.
* A **mask** marks unreliable bits: the top sector (upper eyelid/eyelashes) and the outer quarter of the ring.
* **Hamming distance** = fraction of bits that differ, counted only where both masks are valid. The second
  code is shifted ±16 columns (±11°) and the minimum is taken → tolerance to head tilt.
* Why it works: two *different* irises give essentially random bits → ~50 % disagreement (HD ≈ 0.45–0.5).
  The same iris gives HD ≈ 0.1–0.3.

### 5.6 ORB comparison (3.6) — `compare_iris_orb()`
The lab's example, applied to the segmented crops: ORB finds up to 1000 corner-like keypoints with binary
descriptors; `BFMatcher(NORM_HAMMING).knnMatch(k=2)` finds the two nearest descriptors and **Lowe's ratio
test** (`best < 0.75 × second best`) keeps only unambiguous matches. Score = number of good matches,
MATCH if ≥ 15.

### 5.7 Experiments (3.7–3.9)
* **Mandatory 10 pairs → `iris_results.csv`:** 5 genuine pairs are *cross-session* (image_01 vs image_03 —
  the realistic, harder case), 5 impostor pairs compare neighbouring identities. Columns: images,
  `ground_truth`, `hamming_distance`, `decision`, `correct`, plus the ORB score/decision for comparison.
  Demo result: 9/10 correct with IrisCode (pair 2 falsely rejected, HD 0.377), 10/10 with ORB.
* **All 105 pairs → `results/iris/iris_all_pairs.csv`:** genuine HD 0.142–0.381, impostor HD 0.330–0.499.
  Threshold 0.32 was set just below the closest impostor → FAR 0 %, FRR 13.3 % (2/15), accuracy 98.1 %.
  ORB (≥ 15): accuracy 98.1 %, FAR 2.2 %, FRR 0 % — ORB lets strangers through, IrisCode doesn't.
* **Error analysis (3.9):** the worst genuine pair (iris_02) is shown with its circles: the iris circle of
  `image_02` is too large and off-centre (it reaches the lower eyelid), so the unrolled strip contains
  eyelid instead of iris texture → too many differing bits. Lesson: **segmentation is the weak link**.
  ORB looked good here partly because it also matches eyelashes and the pupil edge, which are not
  unique biometric features — it would not scale to thousands of people.

---

## 6. Assignment 4 — Fingerprint recognition

### 6.1 Data
SOCOFing has one real scan per finger (96×103 px, 500 dpi) plus synthetic alterations.
`image_01` = real, `image_02` = "CR" (central rotation), `image_03` = "Zcut". The lab explicitly allows
"two images or *variants*" of each finger.

### 6.2 Preprocessing (4.2) — `fp_preprocess(path)`
| Step | Code | Why |
|---|---|---|
| Upscale ×3 | `cv2.resize(..., INTER_CUBIC)` | 96 px is too small for ORB's 31-px patches |
| Gaussian blur | `GaussianBlur((5,5))` | removes sensor noise / pores |
| CLAHE | `clahe.apply` | equal ridge contrast across dry/wet areas |
| Otsu binarisation | `THRESH_BINARY_INV + THRESH_OTSU` | Otsu picks the threshold that best separates two intensity groups; INV makes ridges white |
| Clean-up | opening + drop blobs < 60 px | dust and pores would create fake ridge endings |
| Skeleton | `skeletonize` | ridges become 1-pixel lines |
| Mask | closing + erosion of the binary | marks the finger area; outside it, ridges are "cut", not real minutiae |

### 6.3 Minutiae — `extract_minutiae(fp)`
**Crossing number (CN):** walk around the 8 neighbours of a skeleton pixel and count 0→1 transitions.
CN = 1 → **ridge ending**; CN = 3 → **bifurcation** (ridge splits in two); CN = 2 → ordinary ridge.
Raw CN gives hundreds of false minutiae (spurs, broken ridges), so we keep only points inside the mask,
away from the image border and at least 9 px from any other minutia. Result: 24–55 minutiae per print —
the realistic range. Minutiae are visualised and counted (columns `minutiae_1/2`) for analysis; the
decision itself uses ORB + RANSAC.

### 6.4 Matching (4.3) — `compare_fingerprints(p1, p2)`
1. ORB (1500 keypoints) on both enhanced images, k-NN matching, ratio test → `orb_score` (the lab's method,
   MATCH if ≥ 20).
2. **RANSAC geometric verification (our addition):** `cv2.estimateAffinePartial2D(..., method=RANSAC)`
   tries to find *one* rotation + shift + scale that maps image 1's keypoints onto image 2's. Matches that
   agree with that single transform are **inliers**. Same finger placed again → hundreds of inliers.
   Different fingers → their accidental matches point in random directions → 2–4 inliers.
   MATCH if inliers ≥ 12.

Why add RANSAC? In the demo, impostor pairs reached up to **17** ORB good matches — dangerously close
to the lab's threshold of 20. Their RANSAC inliers never exceeded **4**. A bigger safety margin = fewer
false acceptances on new data.

### 6.5 Experiments (4.4–4.6)
* **Mandatory 10 pairs → `fingerprint_results.csv`:** genuine = real vs CR variant; impostor = neighbouring
  fingers. 10/10 correct with both methods. Genuine scores are huge (600–1000) because the variants are
  edits of the *same scan*.
* **All 105 pairs:** 100 % with both methods.
* **Stress test (simulated re-capture):** because SOCOFing is optimistic, we rotate, crop, add noise and
  smudge the real print at 4 levels and match it against the original:

  | Level | Rotation | Area kept | Mean inliers | Accepted |
  |---|---|---|---|---|
  | mild | 5° | 95 % | 505 | 100 % |
  | medium | 15° | 80 % | 351 | 100 % |
  | hard | 30° | 65 % | 49 | 100 % |
  | extreme | 45° | 50 % | 8 | 40 % |

  Lesson: small partial, noisy, smudged prints cause **false rejections** — exactly what happens with a
  dirty sensor or a quick touch.

---

## 7. Section 5 — Comparative analysis

The table (`results/comparison/comparative_analysis.csv/.md`) is generated from the variables computed
above (`FACE_ACCURACY`, `IRIS_ACC`, `FP_ACC`, …), so when you re-run with real photos it updates itself.
It fills in the lab's 7 criteria for each modality. `accuracy_comparison.png` shows the headline numbers.

Section 6 summarises the security side: biometrics cannot be revoked like passwords, templates stay in
memory, `share=True` is avoided, and the main threats (presentation attacks, template theft, poisoning,
adversarial examples) with their mitigations.

---

## 8. Where we improved on the lab's example code

| Lab example | Problem | What we do |
|---|---|---|
| `DeepFace.represent(img_path=face_image, ...)` with the RGB face from `extract_faces` | DeepFace treats arrays as **BGR** → red/blue swapped → embeddings of the group photo don't match registration embeddings as well | Convert RGB → BGR before `represent` |
| Registration via `represent(img_path=path)`, recognition via `extract_faces`+`represent` | Two different code paths can preprocess differently | One `detect_and_embed()` for both |
| `enforce_detection=False` in recognition | If no face is found, DeepFace returns the whole image as a "face" | `enforce_detection=True` + catch → 0 faces |
| Fixed threshold 0.30 | Lab requires experimental justification | Genuine/impostor distances, FAR/FRR sweep, plot |
| `face_results.csv` by hand | Error-prone | Batch evaluation + live *Evaluation* panel |
| ORB on raw iris image | ORB matches eyelashes/eyelids, not iris texture | Segmentation → rubber sheet → IrisCode; ORB kept for comparison |
| ORB only for fingerprints | Impostors reach 17 vs threshold 20 | ORB + RANSAC inliers |
| Unzip anything | Zip-slip, `__MACOSX` junk, different ZIP layouts | `safe_extract()` |
| Phone photos | EXIF rotation ignored → sideways faces → no detection | `ImageOps.exif_transpose` |

---

## 9. Helper scripts

* **`scripts/prepare_open_datasets.py`** — `git clone --filter=blob:none --sparse` downloads only the ~30
  needed files from the GitHub mirrors and writes them in the lab's folder layout (`iris_XX/image_YY.jpg`,
  `finger_XX/image_YY.bmp`). Iris images are saved as JPEG with quality 100 to preserve texture.
* **`scripts/make_demo_faces.py`** — builds the mock team dataset from Faces94. Faces94 has no pose or light
  change, so it **simulates** them: a perspective warp imitates a head turn (`photo_02/03`), gamma + colour
  cast imitates different lighting (`photo_04`), and five synthetic group photos vary brightness, scale,
  head tilt, sensor noise and motion blur, with 0–1 unregistered "strangers". It also writes the matching
  `ground_truth.csv`. Mock data is much easier than real photos (same session, same camera), which is
  why the demo face accuracy is 100 %.

---

## 10. Defense Q&A — the 15 review questions

**1. What is biometric identification?**
Establishing who a person is from measurable physical (face, iris, fingerprint) or behavioural (voice,
gait, typing rhythm) traits. The trait is converted into a numeric template and compared with stored templates.

**2. Identification vs verification?**
Verification is **1:1** — "Is this Aidana?" (compare with one template, answer yes/no; our iris and
fingerprint pairs). Identification is **1:N** — "Who is this?" (search all templates; our face app).
Identification is harder: with N people there are N chances for a false match.

**3. Face detection vs face recognition?**
Detection answers *where is a face* (boxes; it doesn't know who). Recognition answers *whose face is it*
(compares embeddings). Our pipeline does detection first (OpenCV detector), then recognition (FaceNet512).

**4. What is a face embedding?**
A fixed-length vector (512 numbers for FaceNet512) produced by a neural network such that the same person's
photos give nearby vectors and different people give distant vectors. It's the face's "coordinates" in a
learned space; comparing faces = measuring distances.

**5. How are multiple faces recognised in one image?**
The detector returns one box per face; each box is cropped, aligned and embedded separately, and each
embedding is matched against the database independently (`for face in faces:` loop in
`recognize_uploaded_photo`). Counts of detected/recognised/unknown are summed.

**6. Why separate registration and test images?**
To measure performance on *unseen* data. If the test photo were registered, the system would compare a
photo with itself (distance ≈ 0) and report fake 100 % accuracy — like grading a student on the exact
questions they were given the answers to.

**7. Purpose of a similarity threshold?**
It converts a continuous score into a decision and controls the trade-off: lower threshold (stricter) →
fewer false acceptances, more false rejections; higher → the opposite. It's tuned on genuine/impostor score
distributions (our section 2.5, 3.8, 4.5).

**8. What does UNKNOWN mean?**
The closest registered template is still farther than the threshold: either the person is not registered,
or the image quality (blur, dark, side view) made the embedding unreliable. It's the system saying
"not confident", which is safer than guessing a name.

**9. Which iris characteristics are used?**
The texture of the iris stroma: crypts, furrows, collarette, freckles, radial and contraction furrows.
It forms randomly in the womb (even identical twins and your two eyes differ) and is stable for life.
We encode it with Gabor phase bits (IrisCode).

**10. Iris preprocessing methods?**
Grayscale, contrast enhancement (CLAHE), denoising (median/Gaussian blur), segmentation of pupil and
limbus (thresholding, Hough circles), eyelid/eyelash masking, normalisation (Daugman rubber sheet).

**11. Ridge patterns and minutiae?**
Ridges are the raised skin lines; their global pattern is a loop, whorl or arch. **Minutiae** are local
ridge events — mainly **ridge endings** and **bifurcations** — with position and direction. A typical full
print has 30–80; matching ~12+ minutiae in the same geometry is classic forensic evidence of identity.

**12. Fingerprint matching methods?**
(a) Minutiae-based: align two minutiae sets and count paired minutiae (the standard; e.g. NIST Bozorth3).
(b) Correlation/image-based: compare pixel patterns after alignment.
(c) Feature/keypoint-based: ORB/SIFT descriptors + geometric verification (our ORB + RANSAC).
(d) Deep learning embeddings (CNNs trained on fingerprints).

**13. False matches and false rejections?**
False match / false acceptance (FAR): a stranger is accepted as someone else — a *security* failure.
False non-match / false rejection (FRR): the real person is rejected — a *usability* failure. Raising the
threshold trades one for the other; the point where they are equal is the EER (Equal Error Rate).

**14. Effect of lighting, angle, quality?**
Lighting changes pixel values and shadows can hide features; large angles hide half the face and change
its geometry; blur/low resolution destroy fine texture. All of these move genuine scores toward the
threshold → more false rejections, and missed detections. Our mitigations: CLAHE, alignment, registering
several conditions (`photo_02..04`), and measuring robustness (fingerprint stress test).

**15. Cybersecurity threats of storing biometric data?**
* **Irrevocability:** you can't reset your face like a password — one leak is permanent.
* **Template theft / database breach** → identity theft, tracking across systems, possible reconstruction of a face from its embedding.
* **Presentation (spoofing) attacks:** printed photo, phone screen replay, 3D mask, gelatin fingerprint.
* **Database poisoning:** an attacker enrols their face under someone else's name.
* **Adversarial examples:** tiny pixel changes that fool the neural network.
* **Privacy / surveillance and legal compliance** (personal-data laws, consent).
Mitigations: encrypt templates at rest and in transit, cancellable biometrics, liveness detection,
strict access control and audit logs, multi-factor or multi-modal authentication, data minimisation (we keep
templates in memory only and never commit photos).
