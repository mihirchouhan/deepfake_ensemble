import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split

# =========================
# Fix Python path
# =========================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from utils.config import BASE_DIR, MAX_SAMPLES, SEED

# =========================
# Paths
# =========================
GGF_REAL = os.path.join(BASE_DIR, "features", "ggf", "real")
GGF_FAKE = os.path.join(BASE_DIR, "features", "ggf", "fake")

FFT_REAL = os.path.join(BASE_DIR, "features", "spectrum", "real")
FFT_FAKE = os.path.join(BASE_DIR, "features", "spectrum", "fake")

GLCM_REAL = os.path.join(BASE_DIR, "features", "glcm", "real")
GLCM_FAKE = os.path.join(BASE_DIR, "features", "glcm", "fake")

OUT_DIR = os.path.join(BASE_DIR, "dataset")
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# Helper
# =========================
def load_features(feature_dir, max_samples):
    files = sorted(os.listdir(feature_dir))[:max_samples]
    feats = []

    for f in files:
        feats.append(np.load(os.path.join(feature_dir, f)))

    return np.array(feats, dtype=np.float32)

# =========================
# Load & fuse features
# =========================
def build_dataset():
    print("Loading REAL features...")
    ggf_r = load_features(GGF_REAL, MAX_SAMPLES)
    fft_r = load_features(FFT_REAL, MAX_SAMPLES)
    glcm_r = load_features(GLCM_REAL, MAX_SAMPLES)

    print("Loading FAKE features...")
    ggf_f = load_features(GGF_FAKE, MAX_SAMPLES)
    fft_f = load_features(FFT_FAKE, MAX_SAMPLES)
    glcm_f = load_features(GLCM_FAKE, MAX_SAMPLES)

    print("Fusing features...")
    X_real = np.concatenate([ggf_r, fft_r, glcm_r], axis=1)
    X_fake = np.concatenate([ggf_f, fft_f, glcm_f], axis=1)

    y_real = np.zeros(len(X_real), dtype=np.int64)
    y_fake = np.ones(len(X_fake), dtype=np.int64)

    X = np.vstack([X_real, X_fake])
    y = np.concatenate([y_real, y_fake])

    print("Final feature shape:", X.shape)  # (10000, 1300)

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=SEED,
        stratify=y
    )

# =========================
# Run
# =========================
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = build_dataset()

    np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUT_DIR, "y_test.npy"), y_test)

    print("\n✅ STEP 6 COMPLETED: Feature fusion & dataset creation")
