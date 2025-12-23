import os
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

IMAGE_SIZE = (64, 64)
MAX_SAMPLES = 5000
