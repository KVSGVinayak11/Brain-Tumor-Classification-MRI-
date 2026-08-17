"""
Evaluate a saved model on the test split and print a classification
report + confusion matrix.

Usage
-----
    python -m src.evaluate --model-path saved_models/vgg16_final.h5
"""

import argparse
import os

from tensorflow.keras.models import load_model

from src import config
from src.data_loader import get_generators
from src.utils import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained brain tumor MRI classifier.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to a saved .h5 model.")
    parser.add_argument("--data-dir", type=str, default=config.DATA_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    train_dir = os.path.join(args.data_dir, "Training")
    test_dir = os.path.join(args.data_dir, "Testing")

    _, _, test_generator = get_generators(train_dir=train_dir, test_dir=test_dir)

    model = load_model(args.model_path)
    evaluate_model(model, test_generator, save_path=os.path.join(config.ASSETS_DIR, "confusion_matrix.png"))


if __name__ == "__main__":
    main()
