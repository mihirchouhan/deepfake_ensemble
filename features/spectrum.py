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

OUT_DIR = os.path.join(BASE_DIR, "features", "spectrum")
OUT_REAL = os.path.join(OUT_DIR, "real")
OUT_FAKE = os.path.join(OUT_DIR, "fake")

os.makedirs(OUT_REAL, exist_ok=True)
os.makedirs(OUT_FAKE, exist_ok=True)

# =========================
# FFT feature function
# =========================
def compute_fft_feature(image, out_size=32):
    """
    image: (H, W) grayscale
    return: flattened FFT magnitude feature (out_size x out_size)
    """
    # 2D FFT
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)

    # Log-magnitude spectrum
    magnitude = np.log1p(np.abs(fft_shift))

    # Center crop (pure NumPy, no resize dependency)
    H, W = magnitude.shape
    cH, cW = H // 2, W // 2
    half = out_size // 2

    cropped = magnitude[
        cH - half : cH + half,
        cW - half : cW + half
    ]

    return cropped.flatten().astype(np.float32)

# =========================
# Dataset processing
# =========================
def process_split(img_dir, out_dir, label):
    img_files = sorted(os.listdir(img_dir))[:MAX_SAMPLES]

    print(f"\nExtracting Spectrum (FFT) features for {label}")

    for img_f in tqdm(img_files):
        out_path = os.path.join(out_dir, img_f)

        # Cache check
        if os.path.exists(out_path):
            continue

        img = np.load(os.path.join(img_dir, img_f))  # (64,64)
        fft_feat = compute_fft_feature(img)
        np.save(out_path, fft_feat)

# =========================
# Run
# =========================
if __name__ == "__main__":
    process_split(IMG_REAL, OUT_REAL, "REAL")
    process_split(IMG_FAKE, OUT_FAKE, "FAKE")

    print("\n✅ STEP 5A COMPLETED: Spectrum (FFT) features extracted")
