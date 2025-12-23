import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import random
from utils.config import BASE_DIR, SEED

random.seed(SEED)

REAL_IMG_DIR = os.path.join(BASE_DIR, "data", "celeba_real", "img_align_celeba")
FAKE_IMG_DIR = os.path.join(BASE_DIR, "data", "fake_kaggle")

OUT_DIR = os.path.join(BASE_DIR, "precomputed_landmarks")
os.makedirs(OUT_DIR, exist_ok=True)

def make_list(img_dir, out_file, n=5000):
    files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    random.shuffle(files)
    files = files[:n]

    with open(out_file, "w") as f:
        for name in files:
            f.write(os.path.join(img_dir, name) + "\n")

make_list(REAL_IMG_DIR, os.path.join(OUT_DIR, "real_list.txt"))
make_list(FAKE_IMG_DIR, os.path.join(OUT_DIR, "fake_list.txt"))

print("Image lists created (5000 real, 5000 fake)")
