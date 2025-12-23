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

OUT_DIR = os.path.join(BASE_DIR, "features", "glcm")
OUT_REAL = os.path.join(OUT_DIR, "real")
OUT_FAKE = os.path.join(OUT_DIR, "fake")

os.makedirs(OUT_REAL, exist_ok=True)
os.makedirs(OUT_FAKE, exist_ok=True)

# =========================
# GLCM utilities
# =========================
def compute_glcm(image, levels=16):
    """
    Compute normalized GLCM for distance=1, angle=0°
    image: (H,W) grayscale [0,1]
    """
    # Quantize image
    img = np.floor(image * (levels - 1)).astype(np.int32)

    glcm = np.zeros((levels, levels), dtype=np.float32)

    H, W = img.shape
    for y in range(H):
        for x in range(W - 1):
            i = img[y, x]
            j = img[y, x + 1]
            glcm[i, j] += 1

    # Normalize
    if glcm.sum() > 0:
        glcm /= glcm.sum()

    return glcm

def glcm_features(glcm):
    """
    Extract 4 GLCM texture features
    """
    levels = glcm.shape[0]
    i, j = np.meshgrid(range(levels), range(levels), indexing='ij')

    contrast = np.sum(glcm * (i - j) ** 2)
    energy = np.sum(glcm ** 2)
    homogeneity = np.sum(glcm / (1.0 + np.abs(i - j)))

    mu_i = np.sum(i * glcm)
    mu_j = np.sum(j * glcm)
    sigma_i = np.sqrt(np.sum(glcm * (i - mu_i) ** 2))
    sigma_j = np.sqrt(np.sum(glcm * (j - mu_j) ** 2))

    if sigma_i * sigma_j == 0:
        correlation = 0
    else:
        correlation = np.sum(
            glcm * (i - mu_i) * (j - mu_j)
        ) / (sigma_i * sigma_j)

    return np.array(
        [contrast, energy, homogeneity, correlation],
        dtype=np.float32
    )

# =========================
# Dataset processing
# =========================
def process_split(img_dir, out_dir, label):
    img_files = sorted(os.listdir(img_dir))[:MAX_SAMPLES]

    print(f"\nExtracting GLCM texture features for {label}")

    for img_f in tqdm(img_files):
        out_path = os.path.join(out_dir, img_f)

        if os.path.exists(out_path):
            continue

        img = np.load(os.path.join(img_dir, img_f))  # (64,64)
        glcm = compute_glcm(img)
        feats = glcm_features(glcm)

        np.save(out_path, feats)

# =========================
# Run
# =========================
if __name__ == "__main__":
    process_split(IMG_REAL, OUT_REAL, "REAL")
    process_split(IMG_FAKE, OUT_FAKE, "FAKE")

    print("\n✅ STEP 5B COMPLETED: GLCM texture features extracted")
