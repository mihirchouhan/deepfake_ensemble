import sys
import os

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)


import cv2
import numpy as np
from tqdm import tqdm
from utils.config import BASE_DIR, IMAGE_SIZE, REAL_DIR, DATA_DIR

# =========================
# Output directory
# =========================
PREPROCESS_DIR = os.path.join(BASE_DIR, "preprocessed_data")
REAL_OUT = os.path.join(PREPROCESS_DIR, "real")
FAKE_OUT = os.path.join(PREPROCESS_DIR, "fake")

os.makedirs(REAL_OUT, exist_ok=True)
os.makedirs(FAKE_OUT, exist_ok=True)

# =========================
# Helper function
# =========================
def process_folder(input_dir, output_dir, label_name):
    files = sorted(os.listdir(input_dir))
    print(f"\nProcessing {label_name} images: {len(files)} found")

    for idx, file in enumerate(tqdm(files)):
        in_path = os.path.join(input_dir, file)
        out_path = os.path.join(output_dir, f"{label_name}_{idx:06d}.npy")

        # 🔒 Skip if already processed
        if os.path.exists(out_path):
            continue

        try:
            img = cv2.imread(in_path)
            if img is None:
                continue

            # Convert to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Resize
            img = cv2.resize(img, IMAGE_SIZE)

            # Normalize [0,1]
            img = img.astype(np.float32) / 255.0

            # Save
            np.save(out_path, img)

        except Exception as e:
            print(f"Error processing {file}: {e}")

# =========================
# Run preprocessing
# =========================
if __name__ == "__main__":

    print("Starting preprocessing...")

    process_folder(
        input_dir=os.path.join(DATA_DIR, "celeba_real\img_align_celeba"),
        output_dir=REAL_OUT,
        label_name="real"
    )

    process_folder(
        input_dir=os.path.join(DATA_DIR, "fake_kaggle"),
        output_dir=FAKE_OUT,
        label_name="fake"
    )

    print("\nPreprocessing complete.")
