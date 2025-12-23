import sys
import os
import dlib
import numpy as np
from tqdm import tqdm

# =========================
# Fix Python import path
# =========================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from utils.config import BASE_DIR

# =========================
# INPUT PATHS
# =========================
# CelebA images live inside img_align_celeba
REAL_IN = os.path.join(BASE_DIR, "data", "celeba_real", "img_align_celeba")

# Kaggle fake images
FAKE_IN = os.path.join(BASE_DIR, "data", "fake_kaggle")

# =========================
# OUTPUT PATHS
# =========================
LANDMARK_DIR = os.path.join(BASE_DIR, "precomputed_landmarks")
REAL_OUT = os.path.join(LANDMARK_DIR, "real")
FAKE_OUT = os.path.join(LANDMARK_DIR, "fake")

os.makedirs(REAL_OUT, exist_ok=True)
os.makedirs(FAKE_OUT, exist_ok=True)

# =========================
# Load dlib models
# =========================
predictor_path = os.path.join(
    BASE_DIR,
    "models",
    "shape_predictor_68_face_landmarks.dat"
)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# =========================
# Landmark extraction
# =========================
def extract_landmarks(input_dir, output_dir, label, max_images=5000):
    files = sorted(os.listdir(input_dir))
    files = [f for f in files if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    print(f"\nExtracting landmarks for {label}: target {max_images} images")

    saved = 0

    for file in tqdm(files):
        if saved >= max_images:
            break

        in_path = os.path.join(input_dir, file)
        out_path = os.path.join(output_dir, f"{label}_{saved:06d}.npy")

        if os.path.exists(out_path):
            saved += 1
            continue

        try:
            # ✅ THIS IS THE KEY FIX
            img = dlib.load_rgb_image(in_path)
        except Exception:
            continue

        # Face detection
        faces = detector(img, 1)
        if len(faces) == 0:
            continue

        # Use largest face
        face = max(faces, key=lambda r: r.width() * r.height())

        shape = predictor(img, face)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])

        if landmarks.shape != (68, 2):
            continue

        np.save(out_path, landmarks)
        saved += 1

# =========================
# Run
# =========================
if __name__ == "__main__":

    extract_landmarks(REAL_IN, REAL_OUT, "real", max_images=5000)
    extract_landmarks(FAKE_IN, FAKE_OUT, "fake", max_images=5000)

    print("\nLandmark extraction completed successfully.")
