import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp

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
# MediaPipe setup
# =========================
mp_face_mesh = mp.solutions.face_mesh

# ✅ EXACT 68-LANDMARK MAPPING (FIXED)
MP_68_IDX = [
    # Jawline (17)
    234, 93, 132, 58, 172, 136, 150, 149, 176,
    148, 152, 377, 400, 378, 379, 365, 397,

    # Right eyebrow (5)
    70, 63, 105, 66, 107,

    # Left eyebrow (5)
    336, 296, 334, 293, 300,

    # Nose (9)
    168, 6, 197, 195, 5, 4, 1, 19, 94,

    # Right eye (6)
    33, 160, 158, 133, 153, 144,

    # Left eye (6)
    362, 385, 387, 263, 373, 380,

    # Mouth outer (12)
    61, 146, 91, 181, 84, 17,
    314, 405, 321, 375, 291, 308,

    # Mouth inner (8)
    78, 95, 88, 178, 87, 14, 317, 402
]

assert len(MP_68_IDX) == 68, "Landmark index list must contain exactly 68 points"

# =========================
# Core extraction function
# =========================
def extract_landmarks(image_dir, out_dir, label):
    print(f"\nStarting MediaPipe landmark extraction for {label}")

    files = [
        f for f in sorted(os.listdir(image_dir))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    saved = 0

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        for fname in tqdm(files):
            if saved >= MAX_IMAGES:
                break

            img_path = os.path.join(image_dir, fname)
            img = cv2.imread(img_path)

            if img is None:
                continue

            h, w = img.shape[:2]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            result = face_mesh.process(img_rgb)

            if not result.multi_face_landmarks:
                continue

            face_landmarks = result.multi_face_landmarks[0].landmark

            coords = []
            for idx in MP_68_IDX:
                lm = face_landmarks[idx]
                coords.append([lm.x * w, lm.y * h])

            coords = np.array(coords, dtype=np.float32)

            # Safety check
            if coords.shape != (68, 2):
                continue

            out_path = os.path.join(out_dir, f"{label}_{saved:06d}.npy")
            np.save(out_path, coords)
            saved += 1

            if saved == 1:
                print(f"✅ FIRST LANDMARK SAVED: {out_path}")

    print(f"✅ {label} completed: {saved} landmarks saved")

# =========================
# Run
# =========================
if __name__ == "__main__":
    extract_landmarks(REAL_IMG_DIR, REAL_OUT, "real")
    extract_landmarks(FAKE_IMG_DIR, FAKE_OUT, "fake")

    print("\n🎉 STEP 3 (MediaPipe) COMPLETED 🎉")
