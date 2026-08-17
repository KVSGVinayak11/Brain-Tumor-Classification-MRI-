"""
Plotting and evaluation helper functions shared across models.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src import config


def plot_training_curves(history, save_path: str = None):
    """Plot loss & accuracy curves for a Keras `History` object."""
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    epochs = range(len(loss))

    plt.figure(figsize=(15, 5))
    plt.style.use("ggplot")

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, "bo-", label="Train Loss")
    plt.plot(epochs, val_loss, "o-", color="orange", label="Val Loss")
    plt.title("Loss")
    plt.xlabel("epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, "bo-", label="Train Accuracy")
    plt.plot(epochs, val_accuracy, "o-", color="orange", label="Val Accuracy")
    plt.title("Accuracy")
    plt.xlabel("epochs")
    plt.legend()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def evaluate_model(model, test_generator, classes: list = config.CLASS_NAMES, save_path: str = None):
    """
    Run predictions on the test generator, print a classification report
    and plot/save the confusion matrix.
    """
    predictions = model.predict(test_generator)
    predicted_categories = np.argmax(predictions, axis=1)
    true_categories = test_generator.classes

    cm = confusion_matrix(true_categories, predicted_categories)
    print(classification_report(true_categories, predicted_categories, target_names=classes))

    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(ticks=np.arange(len(classes)) + 0.5, labels=[c.title() for c in classes], ha="center")
    plt.yticks(ticks=np.arange(len(classes)) + 0.5, labels=[c.title() for c in classes], va="center")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    return cm
