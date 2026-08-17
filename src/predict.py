"""
Run inference on a single MRI image with a saved model.

Usage
-----
    python -m src.predict --model-path saved_models/vgg16_final.h5 --image path/to/scan.jpg
"""

import argparse

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from src import config


def parse_args():
    parser = argparse.ArgumentParser(description="Classify a single brain MRI image.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to a saved .h5 model.")
    parser.add_argument("--image", type=str, required=True, help="Path to the MRI image to classify.")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=config.CLASS_NAMES,
        help="Ordered list of class names matching the model's training generator.",
    )
    return parser.parse_args()


def predict(model_path: str, image_path: str, classes: list = config.CLASS_NAMES):
    model = load_model(model_path)

    img = load_img(image_path, target_size=config.IMAGE_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)[0]
    predicted_class = classes[int(np.argmax(predictions))]
    confidence = float(np.max(predictions)) * 100

    scores = {c: round(float(p) * 100, 3) for c, p in zip(classes, predictions)}
    return predicted_class, confidence, scores


def main():
    args = parse_args()
    predicted_class, confidence, scores = predict(args.model_path, args.image, args.classes)

    print(f"Predicted class: {predicted_class} ({confidence:.2f}% confidence)")
    print("All class scores:")
    for label, score in scores.items():
        print(f"  {label:12s}: {score:6.3f}%")


if __name__ == "__main__":
    main()
