"""
make_demo_faces.py
==================
Creates a MOCK face dataset so the face-recognition pipeline (Assignments 1-2)
can be built and tested BEFORE the team has taken its own photographs.

!!  These are NOT the team's photos.  For the real lab defense you must replace
!!  them with your own photos (see README -> "Adding your own photos").

Source: "Faces94" face database (Dr Libor Spacek, University of Essex), a free
research dataset of volunteers (not public figures), mirrored on GitHub at
github.com/chenyijia1997/Face-datasets. 180x200 colour frames, 20 per person.

What it produces (same layout as the real dataset):
    <out>/student_faces/<Name>/photo_01.jpg ... test_05.jpg   (640x480 RGB JPG)
    <out>/group_photos/group_01.jpg ... group_05.jpg          (synthetic collages)
    <out>/group_photos/ground_truth.csv                       (who is in each photo)

Faces94 has almost no pose/light variation, so we SIMULATE the lab's protocol:
    photo_01 = frontal frame
    photo_02 = mild perspective warp to the right  (~"head turned right")
    photo_03 = mild perspective warp to the left   (~"head turned left")
    photo_04 = darker, warmer lighting (gamma + colour cast)
    test_05  = a different, later frame (never used for registration)

Usage:
    python scripts/make_demo_faces.py --out data
"""

import argparse
import csv
import io
import random
import subprocess
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np

FACES94_REPO = "https://github.com/chenyijia1997/Face-datasets"

# mock team member -> Faces94 subject id
STUDENTS = {"Aidana": "asamma", "Alikhan": "cwchoi", "Aruzhan": "phughe", "Daniyar": "gdhatc"}
# people who are NOT registered (should come out as UNKNOWN)
STRANGERS = ["vstros", "khughe"]

# ground truth for the 5 synthetic group photos (order = left-to-right placement)
GROUPS = {
    "group_01.jpg": ["Aidana", "Alikhan", "Aruzhan", "Daniyar"],
    "group_02.jpg": ["Daniyar", "Aruzhan", "STRANGER:vstros", "Alikhan", "Aidana"],
    "group_03.jpg": ["Alikhan", "Aidana", "Daniyar", "Aruzhan"],
    "group_04.jpg": ["Aruzhan", "STRANGER:khughe", "Daniyar", "Aidana", "Alikhan"],
    "group_05.jpg": ["Daniyar", "Alikhan", "Aidana", "Aruzhan", "STRANGER:vstros"],
}
# per-photo "conditions" so the 5 photos differ in lighting / arrangement / angle
CONDITIONS = {
    "group_01.jpg": dict(gamma=1.0, tilt=0, scale=1.00, noise=0, blur=0),
    "group_02.jpg": dict(gamma=1.7, tilt=0, scale=0.85, noise=0, blur=0),   # dim room
    "group_03.jpg": dict(gamma=0.7, tilt=6, scale=1.10, noise=0, blur=0),   # bright + tilted heads
    "group_04.jpg": dict(gamma=1.0, tilt=-5, scale=0.75, noise=8, blur=0),  # far away + sensor noise
    "group_05.jpg": dict(gamma=1.3, tilt=3, scale=0.95, noise=0, blur=3),   # slight motion blur
}


def load_faces94(tmp: Path) -> dict[str, list[np.ndarray]]:
    """Download the Faces94 zip (sparse clone) and return {subject: [20 BGR frames]}."""
    repo = tmp / "faces94"
    subprocess.run(["git", "clone", "-q", "--depth", "1", FACES94_REPO, str(repo)], check=True)
    wanted = set(STUDENTS.values()) | set(STRANGERS)
    frames: dict[str, dict[int, np.ndarray]] = {s: {} for s in wanted}
    with zipfile.ZipFile(repo / "faces94.zip") as zf:
        for info in zf.infolist():
            parts = Path(info.filename).parts
            if len(parts) < 2 or parts[-2] not in wanted or not parts[-1].endswith(".jpg"):
                continue
            idx = int(parts[-1].split(".")[-2])           # "asamma.7.jpg" -> 7
            data = np.frombuffer(zf.read(info), np.uint8)
            frames[parts[-2]][idx] = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return {s: [f[i] for i in sorted(f)] for s, f in frames.items()}


def adjust_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """gamma > 1 darkens, gamma < 1 brightens."""
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], np.uint8)
    return cv2.LUT(img, lut)


def yaw_warp(img: np.ndarray, direction: int, strength: float = 0.10) -> np.ndarray:
    """Cheap 'head turn' imitation: squeeze one vertical edge (perspective warp)."""
    h, w = img.shape[:2]
    d = strength * h
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    if direction > 0:   # turn right -> right edge appears shorter
        dst = np.float32([[0, 0], [w, d], [w, h - d], [0, h]])
    else:
        dst = np.float32([[0, d], [w, 0], [w, h], [0, h - d]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def to_640x480(img: np.ndarray) -> np.ndarray:
    """Upscale a 180x200 frame and pad it to the lab's minimum 640x480 resolution."""
    face = cv2.resize(img, (432, 480), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((480, 640, 3), 200, np.uint8)  # neutral light-grey background
    canvas[:, 104:536] = face
    return canvas


def make_student_folders(frames, out: Path) -> None:
    for name, subject in STUDENTS.items():
        f = frames[subject]
        folder = out / "student_faces" / name
        folder.mkdir(parents=True, exist_ok=True)
        shots = {
            "photo_01.jpg": f[0],
            "photo_02.jpg": yaw_warp(f[5], +1),
            "photo_03.jpg": yaw_warp(f[10], -1),
            "photo_04.jpg": cv2.addWeighted(adjust_gamma(f[15], 1.6), 1.0,
                                            np.full_like(f[15], (0, 20, 45)), 1.0, 0),
            "test_05.jpg": f[19],
        }
        for fname, img in shots.items():
            cv2.imwrite(str(folder / fname), to_640x480(img), [cv2.IMWRITE_JPEG_QUALITY, 95])


def make_group_photos(frames, out: Path, seed: int = 7) -> None:
    rng = random.Random(seed)
    gdir = out / "group_photos"
    gdir.mkdir(parents=True, exist_ok=True)
    # frames 2,4,8,13,17 are never used for registration (0,5,10,15) or test (19)
    free_frames = [1, 3, 7, 12, 16]
    rows = []
    for gi, (fname, people) in enumerate(GROUPS.items()):
        c = CONDITIONS[fname]
        W, H = 1600, 900
        # vertical gradient background (looks like a wall)
        bg = np.linspace(170, 110, H, dtype=np.uint8)[:, None, None].repeat(W, 1).repeat(3, 2)
        bg = (bg * np.array([1.0, 0.95, 0.9])).astype(np.uint8)
        n = len(people)
        tile_w = int(250 * c["scale"])
        tile_h = int(tile_w * 200 / 180)
        gap = (W - n * tile_w) // (n + 1)
        for k, person in enumerate(people):
            subject = person.split(":")[1] if person.startswith("STRANGER") else STUDENTS[person]
            img = frames[subject][free_frames[gi]]
            img = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_CUBIC)
            if c["tilt"]:
                M = cv2.getRotationMatrix2D((tile_w / 2, tile_h / 2),
                                            c["tilt"] * (1 if k % 2 else -1), 1.0)
                img = cv2.warpAffine(img, M, (tile_w, tile_h), borderMode=cv2.BORDER_REPLICATE)
            x = gap + k * (tile_w + gap)
            y = (H - tile_h) // 2 + rng.randint(-60, 60)   # people at different heights
            bg[y:y + tile_h, x:x + tile_w] = img
        photo = adjust_gamma(bg, c["gamma"])
        if c["noise"]:
            photo = np.clip(photo + np.random.default_rng(seed).normal(0, c["noise"], photo.shape),
                            0, 255).astype(np.uint8)
        if c["blur"]:
            kernel = np.zeros((c["blur"] * 2 + 1,) * 2)
            kernel[c["blur"], :] = 1.0 / kernel.shape[1]          # horizontal motion blur
            photo = cv2.filter2D(photo, -1, kernel)
        cv2.imwrite(str(gdir / fname), photo, [cv2.IMWRITE_JPEG_QUALITY, 92])
        registered = [p for p in people if not p.startswith("STRANGER")]
        rows.append({"photo": fname, "actual_people": n,
                     "registered_names": ";".join(registered),
                     "unregistered_people": n - len(registered)})
    with open(gdir / "ground_truth.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data", help="output root (default: data)")
    args = ap.parse_args()
    out = Path(args.out)
    with tempfile.TemporaryDirectory() as tmp:
        frames = load_faces94(Path(tmp))
    make_student_folders(frames, out)
    make_group_photos(frames, out)
    print(f"Mock dataset written to {out.resolve()}  (students: {', '.join(STUDENTS)})")


if __name__ == "__main__":
    main()
