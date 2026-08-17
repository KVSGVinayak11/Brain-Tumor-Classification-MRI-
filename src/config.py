"""
Central configuration for the Brain Tumor MRI classification project.
Edit these values to point at your local copy of the dataset and to
tweak training hyperparameters.
"""

import os

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# Root folder that contains the "Training" and "Testing" sub-folders from
# the Kaggle "Brain Tumor MRI Dataset"
# (https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
DATA_DIR = os.environ.get("DATA_DIR", "data")
TRAIN_DIR = os.path.join(DATA_DIR, "Training")
TEST_DIR = os.path.join(DATA_DIR, "Testing")

SAVED_MODELS_DIR = os.environ.get("SAVED_MODELS_DIR", "saved_models")
ASSETS_DIR = os.environ.get("ASSETS_DIR", "assets")

# --------------------------------------------------------------------------
# Data / training hyperparameters
# --------------------------------------------------------------------------
IMAGE_SIZE = (150, 150)
IMAGE_SHAPE = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.15
SEED = 0

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]

EPOCHS = {
    "cnn": 30,
    "vgg16": 20,
    "resnet101": 20,
}
