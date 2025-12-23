import os
import sys
import dlib
import numpy as np
from tqdm import tqdm

# =========================
# Fix Python path
# =========================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from utils.config import BASE_DIR

# =========================
# CONFIG
# =========================
MAX_IMAGES = 5000

REAL_IMG_DIR = os.path.join(
    BASE_DIR, "data", "celeba_real", "img_align_celeba"
)
FAKE_IMG_DIR = os.path.join(
    BASE_DIR, "data", "fake_kaggle"
)

OUT_ROOT = os.path.join(BASE_DIR, "precomputed_landmarks")
REAL_OUT = os.path.join(OUT_ROOT, "real")
FAKE_OUT = os.path.join(OUT_ROOT, "fake")

os.makedirs(REAL_OUT, exist_ok=True)
os.makedirs(FAKE_OUT, exist_ok=True)

# =========================
# DLIB MODELS
# =========================
predictor_path = os.path.join(
    BASE_DIR, "models", "shape_predictor_68_face_landmarks.dat"
)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# =========================
# CORE FUNCTION
# =========================
def extract_landmarks(image_dir, out_dir, label):
    print(f"\nStarting landmark extraction for {label}")

    files = [
        f for f in sorted(os.listdir(image_dir))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    saved = 0

    for fname in tqdm(files):
        if saved >= MAX_IMAGES:
            break

        img_path = os.path.join(image_dir, fname)

        try:
            img = dlib.load_rgb_image(img_path)
        except Exception as e:
            continue  # corrupted image

        faces = detector(img, 0)
        if len(faces) == 0:
            continue

        face = max(faces, key=lambda r: r.width() * r.height())
        shape = predictor(img, face)

        landmarks = np.array([[p.x, p.y] for p in shape.parts()])

        if landmarks.shape != (68, 2):
            continue

        out_path = os.path.join(out_dir, f"{label}_{saved:06d}.npy")
        np.save(out_path, landmarks)
        saved += 1

        if saved == 1:
            print(f"✅ FIRST LANDMARK SAVED: {out_path}")

    print(f"✅ {label} completed: {saved} landmarks saved")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    extract_landmarks(REAL_IMG_DIR, REAL_OUT, "real")
    extract_landmarks(FAKE_IMG_DIR, FAKE_OUT, "fake")

    print("\n🎉 STEP 3 FINISHED 🎉")
