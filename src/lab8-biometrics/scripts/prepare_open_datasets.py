"""
prepare_open_datasets.py
========================
Builds the two OPEN datasets used in Assignments 3 and 4 in exactly the folder
layout the lab instructions ask for:

    data/iris_dataset/iris_01/image_01.jpg ... iris_05/image_03.jpg
    data/fingerprint_dataset/finger_01/image_01.bmp ... finger_05/image_03.bmp

Sources (public GitHub mirrors, used for academic / learning purposes only):
  * Iris  : CASIA-IrisV1 subset mirrored in github.com/yj7082126/OpenCV_IrisRecognition
            (original: Chinese Academy of Sciences, Institute of Automation,
             http://biometrics.idealtest.org / https://nlpr-sir.github.io/dataset/casia-irisv4/)
  * Finger: SOCOFing (Sokoto Coventry Fingerprint Dataset) mirrored in
            github.com/angeliquebouchard/SOCOfing
            (original: https://www.kaggle.com/datasets/ruizgara/socofing, CC BY-NC-SA 4.0)

Only a handful of files are downloaded thanks to a *sparse, blob-less* git clone,
so the script takes seconds instead of downloading hundreds of MB.

Usage (from the project root):
    python scripts/prepare_open_datasets.py              # downloads + builds both
    python scripts/prepare_open_datasets.py --only iris  # just one of them
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

IRIS_REPO = "https://github.com/yj7082126/OpenCV_IrisRecognition"
FINGER_REPO = "https://github.com/angeliquebouchard/SOCOfing"

N_IDENTITIES = 5  # the lab requires >= 5 identities with >= 2 images each


def sparse_clone(repo_url: str, patterns: list[str], dest: Path) -> Path:
    """Clone only the files matching `patterns` (git sparse-checkout, no history)."""
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1", "--filter=blob:none",
         "--no-checkout", repo_url, str(dest)],
        check=True,
    )
    subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", "--no-cone", *patterns],
                   check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "-q"], check=True)
    return dest


def build_iris(tmp: Path) -> None:
    """CASIA-IrisV1 naming: SSS_K_N.bmp -> subject SSS, session K, image N.
    We keep 3 images per identity:
      image_01 = session 1, shot 1   (enrolment-like sample)
      image_02 = session 1, shot 2   (same session  -> easy genuine pair)
      image_03 = session 2, shot 1   (other session -> harder genuine pair)
    """
    src = sparse_clone(IRIS_REPO, [f"data/00[1-{N_IDENTITIES}]/*/*.bmp"], tmp / "iris")
    out_root = DATA_DIR / "iris_dataset"
    for i in range(1, N_IDENTITIES + 1):
        subject = f"{i:03d}"
        picks = {
            "image_01.jpg": src / "data" / subject / "1" / f"{subject}_1_1.bmp",
            "image_02.jpg": src / "data" / subject / "1" / f"{subject}_1_2.bmp",
            "image_03.jpg": src / "data" / subject / "2" / f"{subject}_2_1.bmp",
        }
        out_dir = out_root / f"iris_{i:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, path in picks.items():
            img = cv2.imread(str(path))
            if img is None:
                raise FileNotFoundError(path)
            # JPG with max quality so we do not destroy iris texture
            cv2.imwrite(str(out_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    print(f"[iris] built {N_IDENTITIES} identities in {out_root}")


def build_fingerprints(tmp: Path) -> None:
    """SOCOFing provides ONE real print per finger plus synthetic alterations.
    The lab allows 'two images or variants of each fingerprint', so:
      image_01 = Real print
      image_02 = Altered-Easy 'CR'   (central rotation)
      image_03 = Altered-Easy 'Zcut' (z-cut / partial damage)
    We use the right index finger of subjects 1..5.
    """
    patterns = [
        f"Real/[1-{N_IDENTITIES}]__*Right_index_finger.BMP",
        f"Altered/Altered-Easy/[1-{N_IDENTITIES}]__*Right_index_finger_*.BMP",
    ]
    src = sparse_clone(FINGER_REPO, patterns, tmp / "finger")
    out_root = DATA_DIR / "fingerprint_dataset"
    for i in range(1, N_IDENTITIES + 1):
        real = next((src / "Real").glob(f"{i}__*_Right_index_finger.BMP"))
        prefix = real.stem  # e.g. "1__M_Right_index_finger"
        picks = {
            "image_01.bmp": real,
            "image_02.bmp": src / "Altered" / "Altered-Easy" / f"{prefix}_CR.BMP",
            "image_03.bmp": src / "Altered" / "Altered-Easy" / f"{prefix}_Zcut.BMP",
        }
        out_dir = out_root / f"finger_{i:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, path in picks.items():
            shutil.copy(path, out_dir / name)
    print(f"[finger] built {N_IDENTITIES} fingers in {out_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["iris", "finger"], default=None)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        if args.only in (None, "iris"):
            build_iris(tmp)
        if args.only in (None, "finger"):
            build_fingerprints(tmp)


if __name__ == "__main__":
    main()
