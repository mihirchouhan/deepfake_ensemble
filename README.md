# Deepfake Detection using Heterogeneous Feature Ensemble

This repository contains a complete, CPU-only implementation of a deepfake detection system based on heterogeneous feature ensemble learning. The project uses facial geometry, frequency-domain, and texture features combined with a backpropagation neural network for classification.

---

## Features
- Facial landmark extraction using MediaPipe
- Gray Gradient Feature (GGF)
- Frequency Spectrum Feature (FFT)
- Texture Feature (GLCM)
- Feature fusion (1300-dimensional vector)
- BP Neural Network classifier (scikit-learn)
- Fully CPU-based and portable

---

## 🖥 System Requirements
- OS: Windows / Linux / macOS
- Python: 3.10
- RAM: ≥ 8 GB (16 GB recommended)
- GPU: Not required

---

## 📦 Installation (From Scratch)

### 1️ Clone the repository
bash
git clone https://github.com/mihirchouhan/deepfake_ensemble.git
cd deepfake_ensemble

### 2 Create virtual environment
python -m venv venv
venv\Scripts\activate    # Windows

### 3 Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

### 4 Dataset Setup
Create the following structure:

data/
├── celeba_real/
│   └── img_align_celeba/
└── fake_kaggle/


Place:

CelebA real face images in img_align_celeba

GAN-generated fake images in fake_kaggle

### 5 Running the Project (Step-by-Step)
STEP 1: Preprocess images
python preprocessing/preprocess_images.py

STEP 2: Sample dataset
python preprocessing/sample_dataset.py

STEP 3: Landmark extraction (MediaPipe)
python preprocessing/extract_landmarks_mediapipe.py

STEP 4: Gray Gradient Feature (GGF)
python features/ggf.py

STEP 5A: Spectrum Feature (FFT)
python features/spectrum.py

STEP 5B: Texture Feature (GLCM)
python features/glcm.py

STEP 6: Feature fusion
python features/feature_fusion.py

STEP 7: Train & evaluate BP Neural Network
python models_ml/bpnn.py
