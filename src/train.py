"""
Train one of the three models (cnn / vgg16 / resnet101) on the Brain Tumor
MRI dataset and save the trained model + training curves.

Usage
-----
    python -m src.train --model cnn
    python -m src.train --model vgg16 --epochs 15
    python -m src.train --model resnet101 --batch-size 16
"""

import argparse
import os

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src import config
from src.data_loader import get_generators
from src.models import MODEL_REGISTRY
from src.utils import plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Train a brain tumor MRI classifier.")
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        default="cnn",
        help="Which architecture to train.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--data-dir", type=str, default=config.DATA_DIR)
    parser.add_argument("--output-dir", type=str, default=config.SAVED_MODELS_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    epochs = args.epochs or config.EPOCHS[args.model]

    train_dir = os.path.join(args.data_dir, "Training")
    test_dir = os.path.join(args.data_dir, "Testing")

    train_generator, val_generator, test_generator = get_generators(
        train_dir=train_dir,
        test_dir=test_dir,
        batch_size=args.batch_size,
    )

    num_classes = len(train_generator.class_indices)
    print(f"Detected {num_classes} classes: {train_generator.class_indices}")

    build_fn = MODEL_REGISTRY[args.model]
    model = build_fn(num_classes=num_classes)
    model.summary()

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, f"{args.model}_best.h5")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True),
    ]

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
    )

    os.makedirs(config.ASSETS_DIR, exist_ok=True)
    plot_training_curves(history, save_path=os.path.join(config.ASSETS_DIR, f"{args.model}_curves.png"))

    loss, accuracy = model.evaluate(test_generator)
    print(f"[{args.model}] Test Loss: {loss:.5f} | Test Accuracy: {accuracy:.5f}")

    final_path = os.path.join(args.output_dir, f"{args.model}_final.h5")
    model.save(final_path)
    print(f"Model saved to {final_path}")


if __name__ == "__main__":
    main()
