import os
import sys
import numpy as np
from tqdm import tqdm

# =========================
# Fix Python path
# =========================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from utils.config import BASE_DIR, MAX_SAMPLES

# =========================
# Paths
# =========================
IMG_REAL = os.path.join(BASE_DIR, "preprocessed_data", "real")
IMG_FAKE = os.path.join(BASE_DIR, "preprocessed_data", "fake")

LM_REAL = os.path.join(BASE_DIR, "precomputed_landmarks", "real")
LM_FAKE = os.path.join(BASE_DIR, "precomputed_landmarks", "fake")

OUT_DIR = os.path.join(BASE_DIR, "features", "ggf")
OUT_REAL = os.path.join(OUT_DIR, "real")
OUT_FAKE = os.path.join(OUT_DIR, "fake")

os.makedirs(OUT_REAL, exist_ok=True)
os.makedirs(OUT_FAKE, exist_ok=True)

# =========================
# Core GGF function
# =========================
def compute_ggf(image, landmarks):
    """
    image: (H, W) grayscale, normalized [0,1]
    landmarks: (68,2) float coordinates
    return: (272,) GGF feature vector
    """
    H, W = image.shape
    features = []

    for (x, y) in landmarks:
        x = int(round(x))
        y = int(round(y))

        # Boundary check
        if x <= 0 or x >= W - 1 or y <= 0 or y >= H - 1:
            features.extend([0, 0, 0, 0])
            continue

        g_xp = abs(image[y, x + 1] - image[y, x])
        g_xm = abs(image[y, x] - image[y, x - 1])
        g_yp = abs(image[y + 1, x] - image[y, x])
        g_ym = abs(image[y, x] - image[y - 1, x])

        features.extend([g_xp, g_xm, g_yp, g_ym])

    return np.array(features, dtype=np.float32)

# =========================
# Dataset processing
# =========================
def process_split(img_dir, lm_dir, out_dir, label):
    img_files = sorted(os.listdir(img_dir))[:MAX_SAMPLES]
    lm_files = sorted(os.listdir(lm_dir))[:MAX_SAMPLES]

    print(f"\nExtracting GGF features for {label}")

    for img_f, lm_f in tqdm(zip(img_files, lm_files), total=len(img_files)):
        out_path = os.path.join(out_dir, img_f)

        # Cache check
        if os.path.exists(out_path):
            continue

        img = np.load(os.path.join(img_dir, img_f))      # (64,64)
        lm = np.load(os.path.join(lm_dir, lm_f))         # (68,2)

        ggf = compute_ggf(img, lm)
        np.save(out_path, ggf)

# =========================
# Run
# =========================
if __name__ == "__main__":
    process_split(IMG_REAL, LM_REAL, OUT_REAL, "REAL")
    process_split(IMG_FAKE, LM_FAKE, OUT_FAKE, "FAKE")

    print("\n✅ STEP 4 COMPLETED: GGF features extracted")
