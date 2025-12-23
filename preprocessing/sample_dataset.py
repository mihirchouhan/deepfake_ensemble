import os
import sys
import random
import shutil
from tqdm import tqdm

# =========================
# Fix import path
# =========================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from utils.config import BASE_DIR, SEED

# =========================
# Settings
# =========================
NUM_SAMPLES = 5000

PREPROCESS_DIR = os.path.join(BASE_DIR, "preprocessed_data")
SAMPLED_DIR = os.path.join(BASE_DIR, "sampled_data")

REAL_IN = os.path.join(PREPROCESS_DIR, "real")
FAKE_IN = os.path.join(PREPROCESS_DIR, "fake")

REAL_OUT = os.path.join(SAMPLED_DIR, "real")
FAKE_OUT = os.path.join(SAMPLED_DIR, "fake")

os.makedirs(REAL_OUT, exist_ok=True)
os.makedirs(FAKE_OUT, exist_ok=True)

random.seed(SEED)

# =========================
# Sampling function
# =========================
def sample_files(src_dir, dst_dir, label):
    files = sorted(os.listdir(src_dir))
    print(f"\nFound {len(files)} {label} files")

    if len(files) < NUM_SAMPLES:
        raise ValueError(f"Not enough {label} images to sample")

    sampled = random.sample(files, NUM_SAMPLES)

    for f in tqdm(sampled):
        src = os.path.join(src_dir, f)
        dst = os.path.join(dst_dir, f)
        if not os.path.exists(dst):
            shutil.copy(src, dst)

# =========================
# Run sampling
# =========================
if __name__ == "__main__":

    print("Sampling dataset (paper-aligned)...")

    sample_files(REAL_IN, REAL_OUT, "REAL")
    sample_files(FAKE_IN, FAKE_OUT, "FAKE")

    print("\nSampling completed.")
