"""
Data loading, augmentation and generator utilities.
"""

import os

import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src import config


def list_classes(train_dir: str = config.TRAIN_DIR) -> list:
    """Return the class (sub-folder) names found in the training directory."""
    return sorted(os.listdir(train_dir))


def build_dataframe(root_dir: str, classes: list) -> pd.DataFrame:
    """Build a (Image path, Label) dataframe for a given split directory."""
    image_paths, labels = [], []
    for label in classes:
        class_dir = os.path.join(root_dir, label)
        for img_file in os.listdir(class_dir):
            image_paths.append(os.path.join(class_dir, img_file))
            labels.append(label)
    return pd.DataFrame({"Image": image_paths, "Label": labels})


def get_generators(
    train_dir: str = config.TRAIN_DIR,
    test_dir: str = config.TEST_DIR,
    image_size: tuple = config.IMAGE_SIZE,
    batch_size: int = config.BATCH_SIZE,
    validation_split: float = config.VALIDATION_SPLIT,
    seed: int = config.SEED,
):
    """
    Create the train / validation / test Keras ImageDataGenerators.

    Training data is augmented (rotation, shear, brightness, flips).
    Validation and test data are only rescaled.
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=validation_split,
        rotation_range=10,
        brightness_range=(0.85, 1.15),
        width_shift_range=0.002,
        height_shift_range=0.002,
        shear_range=12.5,
        zoom_range=0,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=validation_split)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        seed=seed,
    )

    val_generator = val_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        seed=seed,
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        seed=seed,
    )

    return train_generator, val_generator, test_generator
